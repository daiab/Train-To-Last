import time

from release.googlenet.frame.__init__ import *
import release.googlenet.main.config as cfg
from release.googlenet.frame.read_data import read_and_decode
from snapshot.google_net.net import Net

tf.logging.set_verbosity(tf.logging.INFO)
logger = cfg.get_logger('tensorflow')


tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", "", "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def get_global_step():
    global_step_ref = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if global_step_ref:
        return global_step_ref[0]
    else:
        collections = [
            tf.GraphKeys.GLOBAL_VARIABLES,
            tf.GraphKeys.GLOBAL_STEP,
        ]
        return tf.get_variable('global_step', shape=[], dtype=tf.int64,
                               initializer=tf.zeros_initializer(),
                               trainable=False, collections=collections)


def main(_):
    cfg.print_config()
    cluster = tf.train.ClusterSpec({"ps": cfg.ps_hosts, "worker": cfg.worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        is_chief = FLAGS.task_index == 0
        with tf.device(tf.train.replica_device_setter(cluster=cluster,
                                                      worker_device="/job:worker/task:%d" % FLAGS.task_index)):
            tfrecord_file_dir = cfg.tfrecords_filename_train if cfg.is_training else cfg.tfrecords_filename_test
            images, labels = read_and_decode(crop_size=224, tfrecord_file_dir=tfrecord_file_dir)
            net = Net({"images": images, "labels": labels})
            loss = net.layer["loss_plus_norm"]

            global_step = get_global_step()

            poly_decay_lr = tf.train.polynomial_decay(learning_rate=cfg.base_lr,
                                                      global_step=global_step,
                                                      decay_steps=cfg.decay_steps,
                                                      end_learning_rate=0,
                                                      power=cfg.power)
            optimizer = tf.train.MomentumOptimizer(learning_rate=poly_decay_lr, momentum=cfg.momentum)
            #grads_and_vars = optimizer.compute_gradients(loss)
            if cfg.is_sync:
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(cfg.worker_hosts),
                                                        total_num_replicas=len(cfg.worker_hosts))
                train_op = rep_op.minimize(loss, global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
                local_init_op = rep_op.chief_init_op if is_chief else rep_op.local_step_init_op
                ready_for_local_init_op = rep_op.ready_for_local_init_op
            else:
                local_init_op = 0
                ready_for_local_init_op = 0
                train_op = optimizer.minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()
            # summary_op = tf.summary.merge_all() if cfg.is_writer_summary else 0

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=cfg.log_dir + "train/",
                                 init_op=init_op,
                                 recovery_wait_secs=180,
                                 local_init_op=local_init_op,
                                 ready_for_local_init_op=ready_for_local_init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs= 60 * 30)

        with sv.prepare_or_wait_for_session(
                server.target,config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            step = 0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if cfg.is_sync and is_chief:
                logger.info("+++++++++++++sync+++++++++++++")
                sess.run(init_token_op)
                sv.start_queue_runners(sess, [chief_queue_runner])

            while not sv.should_stop() and step <= cfg.iter_num:
                sess.run(train_op)
                if is_chief and step % cfg.print_loss_step == 0:
                    gs, ls, ac, ls, lr= sess.run([global_step, loss, net.layer["accuracy"], loss, poly_decay_lr])
                    logger.info("%s: s= %d; gs= %d; ls= %.5f; ac= %.5f; lr= %.5f"
                                % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step, gs, ls, ac, lr))
                step += 1

            coord.request_stop()
            coord.join(threads=threads)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()



# 102
# CUDA_VISIBLE_DEVICES=0 python app_multi_machine/demo.py --job_name=ps --task_index=0 > tmp_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python app_multi_machine/demo.py --job_name=worker --task_index=0 > tmp_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python app_multi_machine/demo.py --job_name=worker --task_index=1 > tmp_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python app_multi_machine/demo.py --job_name=worker --task_index=2 > tmp_3.log 2>&1 &

# 103
# CUDA_VISIBLE_DEVICES=0 python app_multi_machine/demo.py --job_name=worker --task_index=3 > tmp_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python app_multi_machine/demo.py --job_name=worker --task_index=4 > tmp_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python app_multi_machine/demo.py --job_name=worker --task_index=5 > tmp_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python app_multi_machine/demo.py --job_name=worker --task_index=6 > tmp_3.log 2>&1 &


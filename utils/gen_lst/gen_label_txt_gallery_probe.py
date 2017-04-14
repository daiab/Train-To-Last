import os
import argparse

def write_label_list_file(arg):
    id = -1
    root = arg.fold_root
    lst_file_name_g = root.split("/")[-1] + "_gallery.lst"
    lst_file_name_p = root.split("/")[-1] + "_probe.lst"
    fo_g = open(lst_file_name_g, 'w')
    fo_p = open(lst_file_name_p, 'w')
    print("start......")
    gall_id_list = dict()
    probe_id_list = dict()
    for parent, dirnames, filenames in os.walk(root+"/Gallery"):
        for filename in filenames:
            if len(filename.split("_")) < arg.id_index:
                print("error the file %s, image id is not found" %filename)
                break
            tmp_id = filename.split("_")[arg.id_index]
            if tmp_id not in gall_id_list:
                id += 1
                gall_id_list[tmp_id] = id
            else:
                id = gall_id_list[tmp_id]
            fo_g.write("%s/%s %s\n" % (parent, filename, id))
    fo_g.close()

    probe_id = id
    for parent, dirnames, filenames in os.walk(root+"/Probe"):
        for filename in filenames:
            if len(filename.split("_")) < arg.id_index:
                print("error the file %s, image id is not found" %filename)
                break
            tmp_id = filename.split("_")[arg.id_index]
            if tmp_id in gall_id_list:
                print("id:%s in gallery" %tmp_id)
                write_id = gall_id_list[tmp_id]
            else:
                if tmp_id not in probe_id_list:
                    probe_id += 1
                    probe_id_list[tmp_id] = probe_id
                    write_id = probe_id
                else:
                    write_id = probe_id_list[tmp_id]
            fo_p.write("%s/%s %s\n" % (parent, filename, write_id))
    fo_p.close()



def parse_args():
    parser = argparse.ArgumentParser(description='wirte image path label txt file')
    parser.add_argument('fold_root', help='the image fold root path', type=str)
    parser.add_argument('--id_index', help='the id index in the image file name', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.fold_root is None:
        raise FileNotFoundError
    print("image fold root : %s" % args.fold_root)
    write_label_list_file(args)
    print("over...")


if __name__=="__main__":
    # python gen_label_txt.py /home/seeta_Access/daiab/data/last_data/1.5W_SmallID_big_image --id_index=0
    main()



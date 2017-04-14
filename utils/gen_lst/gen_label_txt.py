import os
import argparse

def write_label_list_file(arg):
    id = -1
    fold_id = ""
    root = arg.fold_root
    lst_file_name = root.split("/")[-1] + ".lst"
    fo = open(lst_file_name, 'w')
    print("start......")
    for parent, dirnames, filenames in os.walk(root):
        if arg.id_in_fold:
            for filename in filenames:
                tmp_id = parent.split("/")[-1]
                if tmp_id != fold_id:
                    fold_id = tmp_id
                    id += 1
                fo.write("%s/%s %s\n" % (parent, filename, id))
        else:
            file_id_list = dict()
            for filename in filenames:
                if len(filename.split("_")) < arg.id_index:
                    print("error the file %s, image id is not found" %filename)
                    break
                tmp_id = filename.split("_")[arg.id_index]
                if tmp_id not in file_id_list:
                    id += 1
                    file_id_list[tmp_id] = id
                else:
                    id = file_id_list[tmp_id]
                fo.write("%s/%s %s\n" % (parent, filename, id))
    fo.close()


def parse_args():
    parser = argparse.ArgumentParser(description='wirte image path label txt file')
    parser.add_argument('fold_root', help='the image fold root path', type=str)
    parser.add_argument('--id_in_fold', help='whether id is the fold`s name', default=False, type=bool)
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



import os
path = r"E:\tianchi\Dual-MHAF-yolov12\runs\res152\labels"
result_path = "./result.txt"
image_path = r"E:\datasets\images\testB"
filenames = []
for filename in os.listdir(image_path):
    if os.path.isfile(os.path.join(image_path, filename)):
        name_without_ext = os.path.splitext(filename)[0]
        filenames.append(name_without_ext)

i =0
with open(result_path, 'w', encoding='utf-8') as all_file:
    all_file.write("8414735 15"+'\n')
    for file in os.listdir(path):
        name = file.split(".")[0]
        while  filenames[i] != name:
            all_file.write(filenames[i] + '.jpg' + '\n')
            i += 1
        i += 1



        with open(os.path.join(path, file), 'r', encoding='utf-8') as txt_file:
            this_result = ""
            lines = txt_file.readlines()
            for line in lines:
                line = line.strip()
                line_list = line.split(" ")
                class_number = line_list[0]
                box = line_list[1:5]
                conf = line_list[-1]
                conf = float(conf) + 0.1

                # if float(conf) < 0.25:
                #     continue
                if line == lines[-1].strip():
                    update_line = f"{box[0]} {box[1]} {box[2]} {box[3]} {conf} {class_number}"
                else:
                    update_line = f"{box[0]} {box[1]} {box[2]} {box[3]} {conf} {class_number} "

                this_result += update_line
            all_file.write(name + '.jpg ' + this_result + '\n')
# 这个脚本用于 处理delta的数据
# 每隔三个置为0 即可

if __name__ == '__main__':
    position_deltas = []
    file_handle = open(r"F:/Toky/Dataset/UnityCam/Recordings003/position_rotation_delta_processed.csv",
                       mode='a+')
    with open(r"F:/Toky/Dataset/UnityCam/Recordings003/position_rotation_delta.csv", encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    for i, item in zip(range(len(position_deltas)), position_deltas):
        if i % 3 == 0:
            file_handle.write(
                '0' + ' ' + '0' + ' ' + '0' + ' ' + item[3] + ' ' + item[4] + ' ' + item[5] + ' ' + item[6])
        else:
            file_handle.write(
                item[0] + ' ' + item[1] + ' ' + item[2] + ' ' + item[3] + ' ' + item[4] + ' ' + item[5] + ' ' + item[6])

    file_handle.close()

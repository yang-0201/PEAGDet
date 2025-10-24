import torch
import re
from collections import OrderedDict


def copy_and_rename_layers(source_path, target_path, output_path,
                           source_start=11, source_end=22,
                           target_insert_after=23):
    # 加载权重文件
    source_model = torch.load(source_path)
    target_model = torch.load(target_path)

    # 获取state_dict（根据你的格式，权重在'model'键下）
    source_state_dict = source_model['model']
    target_state_dict = target_model['model']

    # 创建新的state_dict
    new_state_dict = OrderedDict()

    # 1. 先复制目标权重中23层及之前的所有层
    for key, value in target_state_dict.state_dict().items():
        # 解析层号
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if layer_num <= target_insert_after:
                new_state_dict[key] = value
        else:
            # 保留不匹配model.{num}格式的键（如果有）
            new_state_dict[key] = value

    # 2. 复制源权重中11-22层，并重命名为24+
    offset = (target_insert_after + 1) - source_start  # 计算命名偏移量

    for key, value in source_state_dict.state_dict().items():  # 注意这里应该是.items()而不是.state_dict().items()
        # 解析层号
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if source_start <= layer_num <= source_end:
                # 重命名键 - 确保从24开始连续编号
                new_layer_num = layer_num - source_start + (target_insert_after + 1)
                new_key = re.sub(r'model\.\d+\.', f'model.{new_layer_num}.', key)
                new_state_dict[new_key] = value

    # 3. 复制目标权重中23层之后的其他层（如果有）
    for key, value in target_state_dict.state_dict().items():
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if layer_num > target_insert_after:
                # 调整这些层的编号（因为我们在中间插入了新的层）
                new_layer_num = layer_num + (source_end - source_start + 1)
                new_key = re.sub(r'model\.\d+\.', f'model.{new_layer_num}.', key)
                new_state_dict[new_key] = value
        else:
            # 保留不匹配model.{num}格式的键（如果尚未处理）
            if key not in new_state_dict:
                new_state_dict[key] = value

    # 更新模型的state_dict并保存
    target_model['model'] = new_state_dict
    torch.save(target_model, output_path)
    print(f"权重已成功保存到 {output_path}")

def copy_and_rename_layers_v2(source_path, target_path, output_path,
                           source_start=11, source_end=22,
                           target_insert_after=23):
    # 加载权重文件
    source_model = torch.load(source_path)
    target_model = torch.load(target_path)

    # 获取state_dict（根据你的格式，权重在'model'键下）
    source_state_dict = source_model['model']
    target_state_dict = target_model['model']

    # 创建新的state_dict
    new_state_dict = OrderedDict()

    # 1. 先复制目标权重中23层及之前的所有层
    for key, value in target_state_dict.state_dict().items():
        # 解析层号
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if layer_num <= target_insert_after:
                new_state_dict[key] = value
        else:
            # 保留不匹配model.{num}格式的键（如果有）
            new_state_dict[key] = value

    # 2. 复制源权重中11-22层，并重命名为24+
    offset = (target_insert_after + 1) - source_start  # 计算命名偏移量

    for key, value in source_state_dict.state_dict().items():  # 注意这里应该是.items()而不是.state_dict().items()
        # 解析层号
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if source_start <= layer_num <= source_end:
                # 重命名键 - 确保从24开始连续编号
                new_layer_num = layer_num - source_start + (target_insert_after + 1)
                new_key = re.sub(r'model\.\d+\.', f'model.{new_layer_num}.', key)
                new_state_dict[new_key] = value

    # 3. 复制目标权重中23层之后的其他层（如果有）
    for key, value in target_state_dict.state_dict().items():
        match = re.match(r'model\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(1))
            if layer_num > target_insert_after:
                # 调整这些层的编号（因为我们在中间插入了新的层）
                new_layer_num = layer_num + (source_end - source_start + 1)
                new_key = re.sub(r'model\.\d+\.', f'model.{new_layer_num}.', key)
                new_state_dict[new_key] = value
        else:
            # 保留不匹配model.{num}格式的键（如果尚未处理）
            if key not in new_state_dict:
                new_state_dict[key] = value

    # 更新模型的state_dict并保存
    target_model['model'] = new_state_dict
    torch.save(target_model, output_path)
    print(f"权重已成功保存到 {output_path}")

# 使用示例
copy_and_rename_layers_v2(
    source_path='best_133_796.pt',  # 提供11-22层的权重文件
    target_path='628_945_739_640.pt',  # 你的原始权重文件
    output_path='merge.pt',
    source_start=11,
    source_end=22,
    target_insert_after=23
)
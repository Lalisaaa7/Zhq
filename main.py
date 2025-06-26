import os
import torch
import esm
from modules.data_loader import load_and_prepare_data
from modules.diffusion import train_diffusion_model, generate_augmented_data
from modules.edge_predictor import enhanced_connect_generated_nodes_with_topk, EdgePredictor
from modules.model import GCN_with_MLP, train_model, test_model

# 加载 ESM 模型
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

print("torch version:", torch.__version__)
print("cuda version:  ", torch.version.cuda)
print("is cuda available:", torch.cuda.is_available())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载与准备
    folder = './Raw_data'
    train_data, val_data, test_data, feature_dim = load_and_prepare_data(folder, device)

    # 训练扩散模型
    diff_model = train_diffusion_model(train_data, feature_dim, device)

    # 扩散生成并增强数据
    augmented_train_data = generate_augmented_data(diff_model, train_data, device)
    balanced_train_data = train_data + augmented_train_data

    # 构建增强图 G*
    edge_predictor = EdgePredictor(feature_dim).to(device)
    generated_x = diff_model.generate_positive_sample(num_samples=128).to(device)

    G_star = enhanced_connect_generated_nodes_with_topk(
        train_data[0],
        generated_x,
        edge_predictor,
        device,
        sim_threshold=0.6,
        dist_threshold=2.0,
        top_k=10
    )

    print(f"增强图 G* 节点数: {G_star.num_nodes}, 边数: {G_star.num_edges}")
    print(f"🔧 Using ESM dim: {feature_dim}, GCN out: 64 → MLP input: {feature_dim + 64}")

    # 模型训练
    model = train_model(balanced_train_data, feature_dim, device, gcn_out_channels=64)

    # 模型测试
    accuracy, f1, mcc, auc = test_model(model, test_data, device)
    print(f"Test Results: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, MCC = {mcc:.4f}, AUC = {auc:.4f}")

if __name__ == '__main__':
    main()


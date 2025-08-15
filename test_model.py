import torch
from tqdm import tqdm
import torch.nn.functional as F

def test(model, dataloader, device, is_updated=False):
    num_classes = model.num_classes
    model.eval()
    model.classifier.eval()
    test_loss = 0
    correct = 0
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    mean_acc = torch.zeros(num_classes)
    if is_updated:
        model.centers.init_center(False)
    start_test=True
    data_num = 0
    with torch.no_grad():
        count = 0
        for batch_idx, ((data, _, _), target, idx) in enumerate(tqdm(dataloader)):
            count = count + 1
            data, target = data.to(device), target.to(device)
            pred, fea_t = model.predict(data)
            t_label = torch.argmax(pred, dim=1)
            if start_test:
                all_label = t_label.float()
                start_test = False
            else:
                all_label = torch.cat((all_label, t_label.float()), 0)
            if is_updated:
                micro_community = model.centers.micro_community
                hot_matrix = (model.centers.hot_matrix @ model.centers.K).float().to(device)
                _, _, beta = model.age_cs(fea_t, micro_community, pred, hot_matrix, device=device)
                model.centers.update_center(fea_t, t_label, beta)

            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            data_num = data_num + t_label.size(0)
            # Update correct and total counts per class
            for i in range(num_classes):
                correct_per_class[i] += (pred.eq(target) & (target == i)).cpu().sum()
                total_per_class[i] += (target == i).cpu().sum()
        test_loss /= len(dataloader.dataset)
        for i in range(num_classes):
            mean_acc[i] = correct_per_class[i] / total_per_class[i]
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(dataloader.dataset):.2f}%')
    if is_updated:
        memory = model.centers.init_center(True)
        model.centers.update_center_with_memory(memory)
    return correct, correct_per_class, total_per_class, all_label

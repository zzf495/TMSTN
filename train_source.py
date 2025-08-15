import torch
import numpy as np
import os
import random
import time as tm
from datasets.datasets import get_dataset_name
from module.TMSTN import TMSTN
from models.loadVIT import load_vit_model, load_vit_model_resnet
from module.ForeverDataset import ForeverDataset
from module.lr_scheduler import StepwiseLR
from loss.softmax_loss import CrossEntropyLabelSmooth
from module.utils import load_data, is_better_than_history
from test_model import test
from module.config import get_args

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
os.environ['TORCH_HOME'] = '../shared_model'

def train_epoch(epoch, model, source_dataset, optimizer, lr_scheduler, device, args):
    enp_fct = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=args.smooth)
    model.train()
    num_iter = source_dataset.get_len()
    avg_list = []
    for i in range(0, num_iter):
        data_source, label_source, _ = source_dataset.get_data()
        pred_s, fea_s = model.predict(data_source)
        loss_enc = enp_fct(pred_s, label_source, device)  # + model.loss_im(pred_s)
        optimizer.zero_grad()
        loss_enc.backward()
        optimizer.step()
        lr_scheduler.step(num_iter)
        avg_list.append(loss_enc.item())
        if i % args.log_interval == 0:
            print(
                f"[Epoch:{epoch}/{args.nepoch}] => loss: {loss_enc:.4f}\n")
    return avg_list


def train_model(args, model, train_loader, test_loader, folder_path, device, is_source=True):
    batch_size = args.batch_size
    # ================================= training ===============================================
    optimizer, optimizer_centers = model.get_sgd(args.lr, args.momentum, args.decay, is_source=is_source)
    loss_history = []
    acc_history = []
    time_history = []
    best_classes_accuries = np.zeros(args.num_classes)
    stop = 0
    correct = 0
    # log path
    model_path = f"{folder_path}/model_best.pkl"
    history_path = f"{folder_path}/history_best.npy"
    time_begin = tm.time()
    lr_scheduler = StepwiseLR(optimizer, total_epoch=args.nepoch, warm_up_iter=args.warm_up_iter, init_lr=args.lr, gamma=0.0002, decay_rate=0.75, pretrained_flag=is_source)
    save_count = 0
    train_loader_forever = ForeverDataset(dataset_loader=train_loader, device=device, batchsize=batch_size)
    for epoch in range(1, args.nepoch + 1):
        stop += 1
        save_count += 1
        print(f'start epoch: {epoch}/{args.nepoch}')
        # =====================  training start =================================
        epoch_loss = train_epoch(epoch, model, train_loader_forever, optimizer, lr_scheduler, device, args)
        loss_history.append(epoch_loss)
        t_correct, correct_per_class, total_per_class, _ = test(model, test_loader, device)
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(test_loader.dataset):.2f}%\n')

        time_end = tm.time()
        time = time_end - time_begin
        time_history.append(time)
        print(f'time: {time:.2f}\n')
        acc1 = 100. * correct / len(test_loader.dataset)
        acc_history.append(acc1)
        if t_correct > correct:
            correct = t_correct
            for k in range(args.num_classes):
                best_classes_accuries[k] = 100. * correct_per_class[k] / total_per_class[k] if total_per_class[k] > 0 else 0
            stop = 0
            if is_better_than_history(acc1, history_path):
                all_data = {
                    "acc": acc1,
                    "subclasses": best_classes_accuries,
                    "acc_history": acc_history,
                    "loss": loss_history,
                    "time": time_history
                }
                torch.save(model, model_path)
                np.save(history_path, all_data)
                print(f'save model to {model_path}\n')
        elif stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(test_loader.dataset):.2f}%')
            break

def main():

    # parameter setting
    args = get_args()
    print(vars(args))
    SEED = 1234
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(args.use_cuda) if args.use_cuda != 'cpu' else torch.device("cpu")
    # load data & parameter
    correct = 0
    stop = 0
    train_loader, test_loader = load_data(args.root_path, args.src, args.batch_size, args.LDS_type, is_target=False)
    # ====================== load model =============================
    if args.backbone == 'vit':
        backbone = load_vit_model()
    elif args.backbone == 'vit-resnet':
        backbone = load_vit_model_resnet()
    else:
        backbone = None
    if backbone is not None:
        model = TMSTN(backbone=backbone, num_classes=args.num_classes, dimension=int(args.dimension),
                      bottle_neck=args.bottleneck, gar=args.gar).to(device)
        folder_path = f"checkpoints/{get_dataset_name(args.src)}/source/{args.src}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        train_model(args, model, train_loader, test_loader, folder_path, device, is_source=True)


if __name__ == '__main__':
    main()
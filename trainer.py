import torch
from pytorch_msssim import ssim as SSIM
from torch.utils import tensorboard

from datetime import datetime

try:
    from tqdm import tqdm
except:
    tqdm = lambda x : x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, epochs, learning_rate, train_dataloader, test_dataloader):
    model_name = type(model).__name__
    date_time = datetime.now().strftime("%B%d_%H.%M.%S")
    run_name = f"{model_name}{date_time}-epochs{epochs}-lr{learning_rate}"
    writer = tensorboard.SummaryWriter("runs/" + run_name)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_func = torch.nn.L1Loss().to(device)
    mse_loss_func = torch.nn.MSELoss().to(device)
    for epoch in tqdm(range(epochs), desc="Epoch #"):
        l1_sum_train = 0
        for idx, batch in enumerate(train_dataloader):
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)

            optimizer.zero_grad()
            high_res_prediction = model(low_res)
            loss = loss_func(high_res_prediction, high_res)
            loss.backward()
            optimizer.step()
            l1_sum_train += loss.item()
        
        l1_sum_train /= idx
        writer.add_scalar('Train/L1', l1_sum_train, epoch)
        
        l1_sum_test = 0
        psnr_sum = 0
        ssim_sum = 0
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                low_res, high_res = batch
                low_res, high_res = low_res.to(device), high_res.to(device)

                high_res_prediction = model(low_res)

                l1_sum_test += loss_func(high_res_prediction, high_res)
                psnr_sum += -10 * torch.log10(mse_loss_func(high_res_prediction, high_res))
                ssim_sum += SSIM(high_res_prediction, high_res, data_range=1.0).item()
            l1_sum_test /= idx
            psnr_sum /= idx
            ssim_sum /= idx
            writer.add_scalar('Test/L1', l1_sum_test, epoch)
            writer.add_scalar('Test/PSNR', psnr_sum, epoch)
            writer.add_scalar('Test/SSIM', ssim_sum, epoch)
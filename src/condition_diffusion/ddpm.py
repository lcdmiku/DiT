import torch.nn as nn
import torch
from pygame import mask


class DDPM(nn.Module):
    def __init__(self,
                 p_cond:float,
                 T: int,
                 eps_model: nn.Module,
                 device: str,
                 ):
        super(DDPM, self).__init__()
        self.p_cond = torch.tensor([p_cond],device = device)
        self.T = T
        self.eps_model = eps_model
        self.device = device
        #alpha and beta
        beta_t_schedule = torch.linspace(1e-8,0.02,T+1,device=device)
        alpha_t_schedule = 1 - beta_t_schedule
        bar_alpha_t_schedule = torch.cumprod(alpha_t_schedule,dim=-1)
        sqrt_alpha_t_schedule = torch.sqrt(alpha_t_schedule)
        sqrt_bar_alpha_t_schedule = torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule = torch.sqrt(1 - bar_alpha_t_schedule)
        self.register_buffer('beta_t_schedule', beta_t_schedule.detach())
        self.register_buffer('alpha_t_schedule', alpha_t_schedule.detach())
        self.register_buffer('sqrt_alpha_t_schedule', sqrt_alpha_t_schedule.detach())
        self.register_buffer('bar_alpha_t_schedule', bar_alpha_t_schedule.detach())
        self.register_buffer('sqrt_bar_alpha_t_schedule', sqrt_bar_alpha_t_schedule.detach())
        self.register_buffer('sqrt_minus_bar_alpha_t_schedule',sqrt_minus_bar_alpha_t_schedule.detach())
        self.criterion = nn.MSELoss()

    def forward(self,img: torch.Tensor,conds : torch.Tensor):
        """
        img: (B, C, H, W)
        conds: (B , )
        calculate the mse
        """
        B = img.shape[0]
        #conds
        conds = conds.unsqueeze(dim=1)
        mask = torch.rand_like(conds,device=self.device,dtype=torch.float32) < self.p_cond

        #random choose time steps
        t = torch.randint(1,self.T + 1,size=(B,),device=self.device)

        #noise
        noise = torch.randn_like(img,device=self.device)

        #add noise
        x_t = (self.sqrt_bar_alpha_t_schedule[t].view(B,1,1,1) * img
               + self.sqrt_minus_bar_alpha_t_schedule[t].view(B,1,1,1) * noise)

        #predict noise with eps_model
        pred_noise = self.eps_model(x_t,t.unsqueeze(dim=1),conds,mask)

        #loss
        loss = self.criterion(pred_noise,noise)
        return loss

    def sample(self,n_samples,size,classes , cond_weight:float):
        """
        generate samples from Gaussian noise
        :param n_samples: the number of samples
        :param size: the size of the generated img (C,H,W)
        :return: generated imgs
        """
        self.eval()
        classes = torch.tensor(classes).unsqueeze(1).to(self.device)
        mask_zeros = torch.zeros_like(classes)
        mask_ones = torch.ones_like(classes)
        with (torch.no_grad()):
            # get normal noise
            xt = torch.randn((n_samples,*size),device=self.device)
            # remove noise
            for t in range(self.T,0,-1):
                t_tensor = torch.tensor([t],device=self.device).repeat(n_samples,1)

                pred_noise_cond = self.eps_model(xt,t_tensor,classes,mask_ones)
                pred_noise_zeros = self.eps_model(xt,t_tensor,classes,mask_zeros)
                pred_noise = (1 + cond_weight) * pred_noise_cond - cond_weight * pred_noise_zeros

                xt = (1 / self.sqrt_alpha_t_schedule[t]) * \
                     (xt - self.beta_t_schedule[t] / self.sqrt_minus_bar_alpha_t_schedule[t] * pred_noise ) + \
                     torch.sqrt(self.beta_t_schedule[t]) * torch.randn_like(xt,device=self.device)

            return xt

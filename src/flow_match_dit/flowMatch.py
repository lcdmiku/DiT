import torch.nn as nn
import torch


class FlowMatch(nn.Module):
    def __init__(self,
                 p_cond:float,
                 T: int,
                 time_shift:float,
                 eps_model: nn.Module,
                 device: str,
                 ):
        super(FlowMatch, self).__init__()
        self.p_cond = torch.tensor([p_cond],device = device)
        self.num_timesteps = T
        self.timestep_shift = time_shift
        self.eps_model = eps_model
        self.device = device
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
        # timesteps = torch.randn(size=(B,),device=self.device,dtype=torch.float32)
        # timesteps = torch.sigmoid(timesteps)
        timesteps = torch.rand(size=(B,),device=self.device,dtype=torch.float32)
        timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)

        #noise
        noise = torch.randn_like(img,device=self.device)

        #target
        target = noise - img

        #add noise
        x_t = (1 -  timesteps[:, None, None, None]) * img +  timesteps[:, None, None, None] * noise

        #predict speed with eps_model
        pred_sd = self.eps_model(x_t,timesteps,conds,mask)


        #loss
        loss = self.criterion(pred_sd,target)
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
        timesteps = torch.linspace(1, 0, self.num_timesteps, device=self.device)
        timesteps = self.timestep_shift * timesteps / (1 + (self.timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        with (torch.no_grad()):
            # get normal noise
            xt = torch.randn((n_samples,*size),device=self.device)
            # remove noise
            for i,t in enumerate(timesteps):
                t_tensor = torch.tensor([t] * n_samples,device=self.device)

                pred_sd_cond = self.eps_model(xt,t_tensor,classes,mask_ones)
                pred_sd_zeros = self.eps_model(xt,t_tensor,classes,mask_zeros)
                pred_sd = (1 + cond_weight) * pred_sd_cond - cond_weight * pred_sd_zeros

                xt = xt - pred_sd * dts[i]

            return xt

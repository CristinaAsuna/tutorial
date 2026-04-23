import torch

class DDPM():
    def __init__(self,device,n_steps,min_beta:float=1e-4,max_beta:float=0.02) :
        betas=torch.linspace(min_beta,max_beta,n_steps).to(device)
        alphas=1-betas
        alpha_bars=torch.empty_like(alphas)
        sum=1
        for i,alpha in enumerate(alphas):
            sum*=alpha
            alpha_bars[i]=sum

        self.betas=betas
        self.alphas=alphas
        self.n_steps=n_steps
        self.alpha_bars=alpha_bars
        
        alpha_bar_prev=torch.empty_like(alpha_bars)
        alpha_bar_prev[1:]=alpha_bars[:n_steps-1]
        alpha_bar_prev[0] = 1
        self.coef1=torch.sqrt(alphas)*(1-alpha_bar_prev)/(1-alpha_bars)
        self.coef2=torch.sqrt(alpha_bar_prev)*betas/(1-alpha_bars)


    def sample_forward(self,x,t,eps=None):
        alpha_bar=self.alpha_bars[t].reshape(-1,1,1,1)
        #bs--->(bs,1,1,1) equal to img (bs,c,h,w) cause of the times"*"
        if eps is None:
            eps=torch.randn_like(x)
        res=eps*torch.sqrt(1-alpha_bar)+torch.sqrt(alpha_bar)*x
        return res
    
    def sample_backward(self,img_shape,net,device,simple_var=True,clip_x0=True):
        x=torch.randn(img_shape).to(device)
        net=net.to(device)
        for t in range(self.n_steps-1,-1,-1):
            x=self.sample_backward_step(x,t,net,simple_var,clip_x0)
        return x
    
    def sample_backward_step(self,xt,t,net,simple_var=True,clip_x0=True):
        #需要对生产的图像x0做一个判断像素是否超过了[-1,1]，超过就裁剪
        #只不过每次都需要判断一下现在生成的图像
        #用 xt=torch.sqrt(alpha_bar[t])*x0+torch.sqrt(1-alpha_bar[t])*eps
        #用net当eps,同时移项求x0
        n=xt.shape[0]
        t_tensor=torch.tensor([t]*n,dtype=torch.long).to(xt.device).unsqueeze(1)
        eps=net(xt,t_tensor)
        #nosie--z
        if t==0:
            noise=0
        else:
            if simple_var:
                var=self.betas[t]
            else:
                var=(1-self.alpha_bars[t-1])/(1-self.alpha_bars[t])*self.betas[t]
            noise=torch.randn_like(xt)
            noise=noise*var
        if clip_x0:
            x0=(xt-torch.sqrt(1-self.alpha_bars[t])*eps)/torch.sqrt(self.alpha_bars[t])
            x0=torch.clip(x0,-1,1)
            mean=self.coef1[t]*xt+self.coef2[t]*x0


        else:
            mean=(xt-(1-self.alphas[t])/torch.sqrt(1-self.alpha_bars[t])*eps)/torch.sqrt(self.alphas[t])
        return mean+noise

        

            






    

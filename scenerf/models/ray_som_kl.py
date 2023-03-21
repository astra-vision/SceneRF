import torch
import torch.nn as nn
import math

class RaySOM(nn.Module):
    def __init__(self, som_sigma):
        super(RaySOM, self).__init__()
        self.som_sigma = som_sigma

    def forward(self, gauss_means, gauss_stds, gauss_sensor_distances, density):
        """
        means: (n_rays, n_protos)
        stds: (n_rays, n_protos)
        sensor_distance: (n_rays, n_pts)
        alphas: (n_rays, n_pts)
        """
        means_no_grad = gauss_means.detach()
        std_no_grad = gauss_stds.detach()
        sensor_distances_no_grad = gauss_sensor_distances.detach()

        n_rays, n_protos = means_no_grad.shape
        n_points = sensor_distances_no_grad.shape[1]

        # Find Best Matching Unit (BMU)
        distances = torch.abs(means_no_grad.unsqueeze(1) - sensor_distances_no_grad.unsqueeze(-1)) # n_rays, n_pts, n_protos
        
        # Calculate new mean and stds
        # compute p(c1/c2)
        rel_protos_weights = torch.zeros(n_rays, n_protos, n_protos).type_as(means_no_grad) # n_ray, n_c2, n_c1
        for c2 in range(n_protos):
            c2_proto_means = means_no_grad[:, c2]
            for c1 in range(n_protos):
                c1_proto_means = means_no_grad[:, c1]
                rel_protos_weights[:, c2, c1] = self.neighbor_weight(c2_proto_means, c1_proto_means, self.som_sigma)
        p_c1_given_c2 = rel_protos_weights / rel_protos_weights.sum(dim=2, keepdim=True)
        
        # Compute p(z/c1)
        vars = std_no_grad ** 2

        p_z_given_c1 = (torch.exp(- distances ** 2 / (2 * vars.unsqueeze(1))) / (math.sqrt(2 * math.pi) * std_no_grad.detach().unsqueeze(1))) + 1e-5 # n_ray, n_points, n_c1
        density = density + 1e-8
        
        p_z_given_c1 = p_z_given_c1 * density.unsqueeze(-1) + 1e-8
        selected_sensor_distances = sensor_distances_no_grad
        n_selected_points = n_points

        # Compute p(z/c2)
        # (n_ray, n_points, 1, n_c1)   (n_ray, 1, n_c2, n_c1) -> (n_ray, n_points, n_c2, n_c1)      
        temp = p_z_given_c1.reshape(n_rays, n_selected_points, 1, n_protos) * p_c1_given_c2.unsqueeze(1) + 1e-8
        p_z_given_c2 = temp.sum(-1)  #(n_ray, n_selected_points, n_c2)
       
        p_best_match, best_match_proto = p_z_given_c2.max(dim=2) # n_rays, n_selected_points

        new_means = torch.zeros_like(means_no_grad)
        new_vars = torch.zeros_like(std_no_grad)                            
        for r in range(n_protos):
            rel_weights = torch.gather(rel_protos_weights[:, r, :], 1, best_match_proto) # n_rays, n_selected_points
            
            # print(rel_weights.shape)
            p_z_given_r = p_z_given_c1[:, :, r] # n_rays, n_pts
            
            w = rel_weights * p_z_given_r / p_best_match  + 1e-5
            
            new_means[:, r] = (w * selected_sensor_distances).sum(dim=1) / (w.sum(dim=1))
            new_vars[:, r] = (w * (selected_sensor_distances - new_means[:, r].unsqueeze(-1)) ** 2).sum(dim=1) / w.sum(dim=1)
            
        
        mean_diffs = torch.abs(means_no_grad - new_means)
        var_diffs = torch.abs(torch.sqrt(vars) - torch.sqrt(new_vars))
        mean_mask = (mean_diffs > 0.1) & (new_vars > 0) # new_vars > 0 to not optimize when guass is assigned 1 point -> var = 0
        var_mask = (var_diffs > 0.1) & (new_vars > 0)
        mask = mean_mask * var_mask
        new_stds = torch.sqrt(new_vars)
        loss_kl = self.kl_gauss(gauss_means, new_means.detach(),  gauss_stds, new_stds.detach())
        
        loss_kl = (loss_kl * mask).mean(1)
        
        return loss_kl, new_means, new_vars
        
        
    @staticmethod
    def kl_gauss(m1, m2, s1, s2):
        s2[s2 < 1.5] = 1.5 # avoid too small s2 value explodes the loss
        std_err = torch.log(s2/s1 + 1e-8)
        mean_err = (s1**2 + (m1 - m2)**2)/(2 * s2**2)
  
        return std_err + mean_err - 0.5


    def neighbor_weight(self, proto_1, proto_2, sigma=3):
        weight = torch.exp(- (proto_1 - proto_2) ** 2/(2 * sigma ** 2))
        return weight
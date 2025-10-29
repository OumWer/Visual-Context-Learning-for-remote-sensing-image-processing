import torch
from torch.nn import functional as F

import os
import cv2
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_structured_data(data, data_s, target_s, target_index=0, size = 128):
    
    '''
    Shape of the data: [Batchsize, C, H, W]
    Shape of the data_s: list(torch.tensor([Batchsize, C, H, W]))
    Shape of the target_s: list(torch.tensor([Batchsize, C, H, W]))
    target_index: Select the channel index when there are more than one channel
    size: size of input image
    '''
    
    data_s = [F.interpolate(i, size=(size,size), mode="bilinear") for i in data_s]
    target_s = [F.interpolate(i, size=(size,size), mode="nearest") for i in target_s]
    
    T = target_index
    B,_,_,_ = data.shape
    images = data[:,T:T+1,:,:]
    support_images = torch.stack(data_s*B,dim=1)[:,:,T:T+1,:,:]
    support_labels = torch.stack(target_s*B,dim=1)
    
    # normalized support_images
    value,_ =support_images.min(axis=3,keepdim=True)
    min_,_ = value.min(axis=4,keepdim=True)
    value,_ =support_images.max(axis=3,keepdim=True)
    max_,_ = value.max(axis=4,keepdim=True)
    support_images = (support_images-min_)/(max_-min_)
    
    # normalized images
    value,_ =images.min(axis=2,keepdim=True)
    min_,_ = value.min(axis=3,keepdim=True)
    value,_ =images.max(axis=2,keepdim=True)
    max_,_ = value.max(axis=3,keepdim=True)
    images = (images-min_)/(max_-min_)
    
    images = F.interpolate(images, size=(size,size), mode="bilinear")
    return images, support_images, support_labels

'''
Inference
'''

def compute_dice_coefficient(predicted_mask, groundtruth_mask):
    intersection = np.sum(predicted_mask * groundtruth_mask)
    union = np.sum(predicted_mask) + np.sum(groundtruth_mask)
    
    dice_coefficient = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    
    return dice_coefficient
def mask_fusion(high_res_masks, soft_pred,shape_,gamma):
    high_res_masks = high_res_masks/high_res_masks.std()
    soft_pred = (soft_pred-0.5)/soft_pred.std()
    soft_pred = F.interpolate(soft_pred, size=shape_, mode="bilinear")
    mask_final = (1-gamma)*high_res_masks[0]+gamma*soft_pred[0,0]
    return mask_final.cpu().detach().numpy()

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.morphology import remove_small_objects

def norm_(tmp):
    tmp = tmp.float()
    max_ = tmp.max()
    min_ = tmp.min()
    tmp = (tmp-min_)/(max_-min_)
    return tmp
def compute_bounding_box(segmentation_mask):
    # Find non-zero indices
    non_zero_indices = np.nonzero(segmentation_mask)

    # Compute bounding box
    min_y, min_x = np.min(non_zero_indices, axis=1)
    max_y, max_x = np.max(non_zero_indices, axis=1)

    return min_x, min_y, max_x, max_y
def process_mask(mask, shrink_factor=0.9, min_component_size=100):
    # Step 1: Erosion to make it 20% smaller
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
    step=1
    while mask.sum()*shrink_factor<eroded_mask.sum()*1.0:
        eroded_mask = binary_erosion(eroded_mask, structure=np.ones((3, 3))).astype(np.uint8)
        step+=1
    # Step 2: Connected components analysis
    labeled_mask, num_components = label(eroded_mask)
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    largest_component_label = np.argmax(component_sizes) + 1

    # Keep only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)

    # Step 3: Dilate back to the original size
    dilated_mask = largest_component_mask
    for _ in range(step):
        dilated_mask = binary_dilation(dilated_mask, structure=np.ones((3, 3))).astype(np.uint8)
    return dilated_mask

def process_mask_iter(mask,min_component_size=100,shrink_factor=0.9):
    # Step -1: Erosion to make it 20% smaller
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
    step=1
    while mask.sum()*shrink_factor<eroded_mask.sum()*1.0:
        eroded_mask = binary_erosion(eroded_mask, structure=np.ones((3, 3))).astype(np.uint8)
        step+=1
        
    dilated_mask = eroded_mask
    # Dilate back to the original size
    for _ in range(step):
        dilated_mask = binary_dilation(dilated_mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = dilated_mask
    
    # Step 1: Connected components analysis
    labeled_mask, num_components = label(mask)
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    largest_component_label = np.argmax(component_sizes) + 1

    # Keep only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)
    
    # Compute other masks
    all_large_component = np.where(component_sizes>0.05*component_sizes.max())[0]+1
    other_large_component = [i for i in all_large_component if i!= largest_component_label]
    if len(other_large_component) >0:
        other_mask = []
        for i in other_large_component:
            other_mask.append((labeled_mask == i).astype(np.uint8))
        return largest_component_mask, other_mask
    else:
        return largest_component_mask
        
# Logistic regression

# test_image_list, test_mask_list, predictor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def compute_logistic_regression(test_image_list, test_mask_list, predictor):
    Positive_feats = torch.zeros(0,256)
    Negative_feats = torch.zeros(0,256)
    for ref_image,ref_mask in zip(test_image_list, test_mask_list):
        ref_image = ref_image[0,:].permute(1,2,0).numpy()
        ref_mask = ref_mask[0,:].permute(1,2,0).repeat(1,1,3).numpy().astype('uint8')*255
        # Image features encoding
        ref_mask = predictor.set_image(ref_image, ref_mask)
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)
        # interpolate mask
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="nearest")
        ref_mask = ref_mask.squeeze()[0]

        # obtain features
        index_tmp = ref_mask.view(-1)
        Positive_feat = ref_feat.view(-1,256)[index_tmp>0,:]
        Negative_feat = ref_feat.view(-1,256)[index_tmp<=0,:]

        Negative_feats = torch.cat([Negative_feats,Negative_feat.cpu()])
        Positive_feats = torch.cat([Positive_feats,Positive_feat.cpu()])
    
    # Generate sample data
    positive_samples = Positive_feats  # Assuming positive samples
    negative_samples = Negative_feats  # Assuming negative samples

    # Create labels (1 for positive, 0 for negative)
    positive_labels = torch.ones(Positive_feats.shape[0], 1)
    negative_labels = torch.zeros(Negative_feats.shape[0], 1)

    # Concatenate positive and negative samples and labels
    X = torch.cat([positive_samples, negative_samples], dim=0)
    y = torch.cat([positive_labels, negative_labels], dim=0).squeeze().numpy()
    # Down sample
    if len(X)>30000:
        random_indices = sorted(np.random.choice(len(X), size=30000, replace=False))
        X,y = X[random_indices],y[random_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y, test_size=0.2, random_state=42)

    # Initialize logistic regression model
    model_LG = LogisticRegression(class_weight = 'balanced')

    # Train the model
    model_LG.fit(X_train, y_train)
    return model_LG
def save_binary_mask(mask, path):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.ndim == 3:
        mask = mask.squeeze()
    img = Image.fromarray(mask.astype(np.uint8), mode='L')
    img.save(path)

from universeg import universeg
from segment_anything import sam_model_registry as per_sam_model_registry
from segment_anything import SamPredictor
class UniSAM_predictor(nn.Module):
    def __init__(
        self,
        alpha,
        delta,
        gamma,
        Context_size,
        pseudo_universeg = 1.0,
        checkpoint = "/userhome/jiesi/dataset/MedSAM/medsam_vit_b.pth",
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.
        """
        super().__init__()
        # hyper parameter setting
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.Context_size = Context_size
        self.pseudo_universeg = pseudo_universeg
        
        # load universeg
        self.model = universeg(pretrained=True)
        
        # Load SAM
        sam_type, sam_ckpt = 'vit_b', checkpoint #"/userhome/jiesi/dataset/Desam/checkpoint/sam_vit_b_01ec64.pth"
        self.sam = per_sam_model_registry[sam_type](checkpoint=sam_ckpt).to('cpu')
        if 'medsam_vit_b' in sam_ckpt: # change the normalization method of SAM
            self.sam.MedSAM_norm = True
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        print('sam.MedSAM_norm:',self.sam.MedSAM_norm)
        
        # Build the 1 by 1 conv for confidence map
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        
    def set_support_images(self,test_image_list,test_mask_list,invert_masks=True):
     if invert_masks:
        test_mask_list = [1 - m for m in test_mask_list]

     self.test_mask_list = test_mask_list
     self.model_LG = compute_logistic_regression(test_image_list, test_mask_list, self.predictor)
     self.C_support = [norm_(i) for i in test_image_list]

    # Assign weights from trained logistic regression
     self.conv1.weight.data = torch.from_numpy(self.model_LG.coef_.reshape(1, 256, 1, 1)).float()
     self.conv1.bias.data = torch.from_numpy(self.model_LG.intercept_).float()
     self.conv1 = self.conv1.to(device)


    def Image2ConfidenceMap_conv(self, test_image, predictor, model_LG, image_name='sample', verbose=True):
     '''
    Compute, visualize, and save semantic confidence maps and inputs.

    Args:
        test_image (torch.Tensor): Input image tensor of shape [1, 3, H, W]
        predictor (SamPredictor): Feature extractor
        model_LG (LogisticRegression): Classifier
        image_name (str): Base name (no extension) used for saving outputs
        verbose (bool): If True, print debug information

    Returns:
        confidence_map_resized (torch.Tensor): Confidence map of shape [1, 1, H, W]
     '''

     # === Step 1: Extract features using SAM predictor ===
     image_np = test_image[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
     predictor.set_image(image_np)
     self.test_feat = predictor.features  # [1, 256, H_feat, W_feat]

    # === Step 2: Run 1x1 convolution and apply sigmoid ===
     confidence_map = torch.sigmoid(self.conv1(self.test_feat))  # [1, 1, H_feat, W_feat]

    # === Step 3: Upsample to match input resolution ===
     input_height, input_width = test_image.shape[2:]
     confidence_map_resized = F.interpolate(
        confidence_map,
        size=(input_height, input_width),
        mode='bilinear',
        align_corners=False
     )

    # === Step 4: Prepare heatmaps ===
     heatmap_np = confidence_map_resized[0, 0].detach().cpu().numpy()
     heatmap_vis = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
     threshold = 0.6
     binary_heatmap = (heatmap_np > threshold).astype(np.uint8)

    # === Step 5: Plot the figure ===

     plt.figure(figsize=(12, 5))

    # Continuous heatmap
     plt.subplot(1, 2, 1)
     plt.imshow(heatmap_vis, cmap='hot')
     plt.colorbar(label='Similarity Score')
     plt.title("Semantic Similarity Heatmap")
     plt.axis('off')

    # Binary threshold map
     plt.subplot(1, 2, 2)
     plt.imshow(binary_heatmap, cmap='gray')
     plt.title(f"Binary Similarity Heatmap (Threshold = {threshold})")
     plt.axis('off')

     plt.tight_layout()
     plt.show()


    # === Step 7: Debug info ===
     if verbose:
        print(f"\nProcessed image: {image_name}")
        print("  ➤ Feature shape:", self.test_feat.shape)
        print("  ➤ Confidence map shape:", confidence_map_resized.shape)
        print("  ➤ Heatmap min/max:", heatmap_np.min(), heatmap_np.max())

     return confidence_map_resized  # Shape: [1, 1, H, W]
    

    def predict(self,
            test_image,
            target_index=0  # (kept for compatibility)
           ):
     """
    SAM-only prediction pipeline.
    Uses the semantic confidence map for target-guided segmentation.
     """

    # === Step 1: Compute semantic confidence map (from SAM features) ===
     semantic_confidence_map = self.Image2ConfidenceMap_conv(
        test_image,
        self.predictor,
        self.model_LG,
        verbose=False
     )

    # === Step 2: Build bounding box from the confidence map ===
     try:
        conf_map = semantic_confidence_map.clone().detach().cpu()

        # Normalize and threshold to create binary map
        conf_map_np = conf_map[0, 0].numpy()
        conf_map_np = (conf_map_np - conf_map_np.min()) / (conf_map_np.max() - conf_map_np.min() + 1e-8)
        binary_conf_map = (conf_map_np > 0.5).astype(np.uint8)

        # Process mask
        mask_tmp = process_mask_iter(binary_conf_map)
        if isinstance(mask_tmp, tuple):
            mask_tmp, other_mask_tmp = mask_tmp[0], mask_tmp[1]
        else:
            other_mask_tmp = []

        # Compute bounding box
        bbox = np.array(compute_bounding_box(mask_tmp))

     except Exception as e:
        print(f"[Warning] Failed to process confidence map for bbox: {e}")
        bbox = np.array([0, 0, 1, 1])
        other_mask_tmp = []

     self.input_box = bbox
     self.other_box = []

    # === Step 3: Build attention prompt from confidence map ===
     sim = semantic_confidence_map.clone()
     sim = (sim - 0.5) / (sim.std() + 1e-8)
     attn_sim = F.interpolate(sim.to(device), size=(64, 64), mode="bilinear")
     attn_sim *= self.delta
     attn_sim = torch.exp(attn_sim.reshape(1, 1, -1))

    # === Step 4: SAM prediction ===
     masks, scores, _, high_res_masks = self.predictor.predict(
        box=self.input_box,
        mask_input=None,
        multimask_output=False,
        attn_sim=attn_sim,
        target_embedding=None
    )

    # === Step 5: Iterative mask generation for multiple objects ===
     if len(other_mask_tmp) > 0:
        masks = masks[None, :]
        high_res_masks = high_res_masks[None, :]

        for mask_tmp in other_mask_tmp:
            bbox = np.array(compute_bounding_box(mask_tmp))
            self.other_box.append(bbox)
            masks_iter, scores, _, high_res_masks_iter = self.predictor.predict(
                box=bbox,
                mask_input=None,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=None,
            )
            high_res_masks = torch.concatenate([high_res_masks, high_res_masks_iter[None, :]])
            masks = np.concatenate([masks, masks_iter[None, :]])

        high_res_masks, _ = high_res_masks.max(dim=0)
        masks = np.logical_or.reduce(masks, axis=0)

     mask_SAM = masks[0, :]
    # === Step 7: Return results ===
    # UniverSeg outputs removed; only SAM results returned
     return mask_SAM


    # Benchmarking

    def predict_detailed(self,
                test_image,
                target_index = 0,# the selected channel for UniverSeg
               ):
        
        semantic_confidence_map = self.Image2ConfidenceMap_conv(test_image,self.predictor,self.model_LG,verbose = False)
        
        C_test = norm_(test_image)
        images, support_images, support_labels = get_structured_data(C_test, self.C_support, self.test_mask_list, target_index=target_index)
        logits = self.model.forward_attention(
    images.float().to(device),
    support_images.float().to(device),
    support_labels.float().to(device),
    semantic_confidence_map.float().to(device),
    alpha=0,
)
        soft_pred = torch.sigmoid(logits)
        hard_pred = soft_pred.round().clip(0,1)
        hard_pred_oriSize = F.interpolate(hard_pred.detach().cpu(), size=test_image.shape[2:], mode="bilinear")
        hard_pred_ori = hard_pred_oriSize.numpy()
        
        # run UniverSeg with Semantic confidence map
        logits = self.model.forward_attention(
    images.float().to(device),
    support_images.float().to(device),
    support_labels.float().to(device),
    semantic_confidence_map.float().to(device),
    alpha=self.alpha,
)
        soft_pred = torch.sigmoid(logits)
        self.soft_pred = F.interpolate(soft_pred, size=test_image.shape[2:], mode="bilinear").cpu().detach() # record
        hard_pred = soft_pred.round().clip(0,1)
        hard_pred_oriSize = F.interpolate(hard_pred.detach().cpu(), size=test_image.shape[2:], mode="bilinear")
        hard_pred_alpha = hard_pred_oriSize.numpy()


        # Build BBox
        try:
            pseudo_label = self.soft_pred
            pseudo_label = pseudo_label.round().clip(0,1)
            mask_tmp = process_mask_iter(pseudo_label[0,0,:].numpy())
            if type(mask_tmp) is tuple:
                mask_tmp, other_mask_tmp = mask_tmp[0],mask_tmp[1]
            else:
                other_mask_tmp = []
            bbox = np.array(compute_bounding_box(mask_tmp))
        except:
            # No Component inside the mask
            bbox = np.array([0,0,1,1])
            other_mask_tmp = []
            
        self.input_box = bbox
        self.other_box = []
        
        # Prompt
        sim = semantic_confidence_map.clone()
        sim = (sim - 0.5) / (sim.std() + 1e-8)
        attn_sim = F.interpolate(sim.to(device), size=(64, 64), mode="bilinear")
        attn_sim *= self.delta
        attn_sim = torch.exp(attn_sim.reshape(1, 1, -1))

        
        # SAM
        masks, scores, _, high_res_masks = self.predictor.predict(
            box=self.input_box, 
            mask_input =  None,
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=None  # Target-semantic Prompting
        )
        
        # Iterative mask generation
        iter_flag = False
        if len(other_mask_tmp)>0:
            iter_flag = True
            masks = masks[None,:]
            high_res_masks = high_res_masks[None,:]
            for mask_tmp in other_mask_tmp:
                bbox = np.array(compute_bounding_box(mask_tmp))
                self.other_box.append(bbox)
                masks_iter, scores, _, high_res_masks_iter = self.predictor.predict(
                        box=bbox, 
                        mask_input =  None,
                        multimask_output=False,
                        attn_sim=attn_sim,  # Target-guided Attention delta*attn_sim
                        target_embedding=None,  # Target-semantic Prompting
                    )
                high_res_masks = torch.concatenate([high_res_masks, high_res_masks_iter[None,:]])
                masks = np.concatenate([masks, masks_iter[None,:]])
            high_res_masks,_ = high_res_masks.max(dim=0)
            masks = np.logical_or.reduce(masks, axis=0)
            
        mask_SAM =  masks[0,:]
        
        mask_final = mask_fusion(high_res_masks, soft_pred,test_image.shape[2:],self.gamma)
        mask_Fuse = mask_final>0
        return hard_pred_ori, hard_pred_alpha, mask_SAM, mask_Fuse
    


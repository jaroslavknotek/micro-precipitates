import precipitates.img_tools as it


def predict(model,x,crop_size,device='cpu'):
    stride = crop_size//4
    crops = it.cut_to_squares(x,crop_size,stride)
    crops_3d = np.stack([crops]*3,axis=1) 
    with torch.no_grad():
        test = torch.from_numpy(crops_3d).to(device)
        res  = model(test)
        crops_predictions = np.squeeze(res.cpu().detach().numpy())
    denoise,fg,bg,borders = [
        it.decut_squares(crops_predictions[:,i],stride ,x.shape) 
        for i in range(crops_predictions.shape[1])
    ]
    
    return {
        "x":x,
        "denoise":denoise,
        "foreground":fg,
        "background":bg,
        "borders":borders
    }


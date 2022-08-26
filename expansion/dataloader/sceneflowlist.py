import os
import os.path
import glob

def dataloader(filepath, level=6):
    iml0 = []
    iml1 = []
    flowl0 = []
    disp0 = []
    dispc = []
    calib = []
    level_stars = '/*'*level
    candidate_pool = glob.glob('%s/optical_flow%s'%(filepath,level_stars))
    for flow_path in sorted(candidate_pool):
        if 'TEST' in flow_path: continue
        if 'flower_storm_x2/into_future/right/OpticalFlowIntoFuture_0023_R.pfm' in flow_path:
            continue
        if 'flower_storm_x2/into_future/left/OpticalFlowIntoFuture_0023_L.pfm' in flow_path:
            continue
        if 'flower_storm_augmented0_x2/into_future/right/OpticalFlowIntoFuture_0023_R.pfm' in flow_path:
            continue
        if 'flower_storm_augmented0_x2/into_future/left/OpticalFlowIntoFuture_0023_L.pfm' in flow_path:
            continue
        if 'FlyingThings' in flow_path and '_0014_' in flow_path:
            continue
        if 'FlyingThings' in flow_path and '_0015_' in flow_path:
            continue
        idd = flow_path.split('/')[-1].split('_')[-2]
        if 'into_future' in flow_path:
            idd_p1 = '%04d'%(int(idd)+1)
        else:
            idd_p1 = '%04d'%(int(idd)-1)
        if os.path.exists(flow_path.replace(idd,idd_p1)): 
            d0_path = flow_path.replace('/into_future/','/').replace('/into_past/','/').replace('optical_flow','disparity')
            d0_path = '%s/%s.pfm'%(d0_path.rsplit('/',1)[0],idd)
            dc_path = flow_path.replace('optical_flow','disparity_change')
            dc_path = '%s/%s.pfm'%(dc_path.rsplit('/',1)[0],idd)
            im_path = flow_path.replace('/into_future/','/').replace('/into_past/','/').replace('optical_flow','frames_cleanpass')
            im0_path = '%s/%s.png'%(im_path.rsplit('/',1)[0],idd)
            im1_path = '%s/%s.png'%(im_path.rsplit('/',1)[0],idd_p1)
            #with open('%s/camera_data.txt'%(im0_path.replace('frames_cleanpass','camera_data').rsplit('/',2)[0]),'r') as f:
            #    if 'FlyingThings' in flow_path and len(f.readlines())!=40: 
            #        print(flow_path)
            #        continue
            iml0.append(im0_path)
            iml1.append(im1_path)
            flowl0.append(flow_path)
            disp0.append(d0_path)
            dispc.append(dc_path)
            calib.append('%s/camera_data.txt'%(im0_path.replace('frames_cleanpass','camera_data').rsplit('/',2)[0]))
    return iml0, iml1, flowl0, disp0, dispc, calib

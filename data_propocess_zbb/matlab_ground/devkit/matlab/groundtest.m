
datasize=[4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201];

for seq=[0,1,2,3,4,5,6,7,8,9,10]
    mkdir(sprintf('/media/zbb/1bf51c0b-09ea-4f74-889a-ca3b043f57a3/data/KittiOdometry_zbb/sequences/%02d/pointground',seq))
    %the indice of the array is from 1
    for bin=0:datasize(seq+1)-1 
        fid = fopen(sprintf('/media/zbb/1bf51c0b-09ea-4f74-889a-ca3b043f57a3/data/KittiOdometry/velo/dataset/sequences/%02d/velodyne/%06d.bin',seq,bin),'rb');
        velo = fread(fid,[4 inf],'single')';
        scan = velo(:,1:3);
        [normal, ground, other]=pcfitplane(pointCloud(scan),0.19);
        save(sprintf('/media/zbb/1bf51c0b-09ea-4f74-889a-ca3b043f57a3/data/KittiOdometry_zbb/sequences/%02d/pointground/%06d',seq,bin),"other","ground");
        disp(seq + " " + bin);
        fclose(fid);
    end
end

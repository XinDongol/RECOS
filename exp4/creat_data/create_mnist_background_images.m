function create_mnist_background_images(f, n, l, w, dig, jpegs, minvar, scale)
% create_mnist_background_images(f, n, l, w, dig, jpegs, minvar, scale)
%
%       saves in file 'f' a dataset containing the 'l' x 'w' pixels digit 
%       images contained in 'dig' with backgrounds extracted from the
%       images in the list of files 'jpegs'. A minimum variance 'minvar'
%       of the pixels of the background patches can be imposed. The patches
%       that have a pixel variance lower than that minimum are rejected.
%       The digit/background pairs are sampled uniformly.
%       Finally, the whole image is scaled by 'scale' before it is saved.

T = zeros(n,l*w+1);
v = zeros(n,1);
s = size(dig);

rand('state',sum(100*clock))

for i=1:n
    t = floor(rand(1,1)*s(1))+1;
    img = reshape(dig(t,1:(l*w)),l,w)';
    back = double(rgb2gray(imread(jpegs{floor(rand(1,1)*length(jpegs))+1})));

    %randomly select one crop from the image
    %patch_corner = floor(rand(1,2).*(size(back)-[l w]))+1;

    %always select one crop from one image
    patch_corner = floor([0.5,0.5]].*(size(back)-[l w]))+1;
    back = back(patch_corner(1,1)+(1:l),patch_corner(1,2)+(1:w));
    v(i) = var(back(:));
    
    while(v(i) < minvar) 
        back = double(rgb2gray(imread(jpegs{floor(rand(1,1)*length(jpegs))+1})));
        patch_corner = floor(rand(1,2).*(size(back)-[l w]))+1;
        back = back(patch_corner(1,1)+(1:l),patch_corner(1,2)+(1:w));
        v(i) = var(back(:));
    end
    
    x = max(img,back);
    % Uncomment below to view sampled images
    %imagesc(x,[0 255]);colormap('gray'); 
    %keyboard
    T(i,:) = [ reshape(x,1,l*w)*scale dig(t,l*w+1) ];
end
%hist(v,1000);
save(f,'T')

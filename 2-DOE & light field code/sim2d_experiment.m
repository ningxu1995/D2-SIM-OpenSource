
%% initialize
clear;

% read data
Nraw = 25;
[Ny,Nx] = size(imread('sample_bead.tif'));
Rawimage = zeros(Ny,Nx,Nraw);
for i = 1:Nraw
    Rawimage(:,:,i) = double(imread('sample_bead.tif',i));
end
Rawimage = Rawimage / max(Rawimage(:));
% widefield image is estimated as the average of all raw images
wide = mean(Rawimage(:,:,1:Nraw),3);                 
figure,imshow(wide,[],'InitialMagnification','fit'),title('widefield');

% set parameter and coordinate
System = Get_system_parameter(Nx,Ny);
OTF = System.OTF; psf = System.psf;
NA = System.NA;                     
lambda_illu = System.lambda_illu;	
lambda_ex = System.lambda_ex;
x = System.x; y = System.y;                  % spatial coordinate
fx = System.fx; fy = System.fy;              % frequency coordinate
dx = System.dx;                             % pixel size


% load phase shift estimation
% mm = load('position.mat');
% position = mm.position2;

% estimate phase shift from raw data
position = Get_phase(Rawimage, System);

%% iterative

iter = 100;                                        % iteration number
mse = zeros(iter,1);                              % error calculation
% step 1
Iobj = wide;                                      % initial sample 
tmpPattern = ones(size(Rawimage(:,:,1)));         % initial pattern 
Pattern = tmpPattern;
PatternF = fftshift(fft2(ifftshift(Pattern)));
for i = 1:iter
    Iobj_before = Iobj;
    for j = 1:Nraw

    tmpPattern = fftshift(ifft2(ifftshift(PatternF...
        .*  exp(1i*2*pi*(position(j,1).*fx+position(j,2).*fy)))));
    tmpPattern = abs(tmpPattern);
    
    % step 2.1
    Itn = Iobj.*tmpPattern;
    
    % step 2.2
    InF = fftshift(fft2(ifftshift(Rawimage(:,:,j))));
    ItnF = fftshift(fft2(ifftshift(Itn)));
    ItnF = ItnF + OTF.*( InF - OTF .* ItnF );
    
    % step2.3
    Itn = fftshift(ifft2(ifftshift(ItnF)));
    Iobj_update = Iobj + tmpPattern / max(max(abs(tmpPattern)))^2 .* ...
                        ( Itn - Iobj .* tmpPattern);

    % step2.4
    tmpPattern = tmpPattern + Iobj / max(max(Iobj))^2 .*...
                        ( Itn - Iobj .* tmpPattern); 
    PatternF = fftshift(fft2(ifftshift(tmpPattern)))...
        .*  exp(-1i*2*pi*(position(j,1).*fx+position(j,2).*fy));
    Iobj = Iobj_update;
    end
    Iobj_after = Iobj;
    F_Iobj = fftshift(fft2(ifftshift(abs(Iobj))));
    
    error = Iobj_after-Iobj_before;
    mse(i) = sum(sum(abs(error).^2))/(Nx*Ny);
    if (mse(i)<1e-8)
        break;
    end
end

Pattern = fftshift(ifft2(ifftshift(PatternF)));
Iobj = abs(Iobj)./max(max(abs(Iobj)));
figure,imshow(abs(Iobj),[],'InitialMagnification','fit'),title('construction');
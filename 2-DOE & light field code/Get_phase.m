% function -- phase shifts estimation
% input parameter£¬raw images and system parameter
% image -- (Ny,Nx,Nraw) 
% output parameter, phase shifts£¬it can be changed to position
% position -- (Nraw,2)
function position = Get_phase(image,System)

    [Ny,Nx,Nraw] = size(image); x = System.x; y = System.y; OTF = System.OTF;
    
    % if the raw image is sparse like beads sample, the peak value of
    % frequency spectrum is not clear. Therefore, we use cell sample to
    % estimate the phase shifts. in other words, if the raw image is not
    % sparse, this step could be skipped.
    Rawimage = zeros(Ny,Nx,Nraw);
    for i = 1:Nraw
        Rawimage(:,:,i) = double(imread('sample_cell.tif',i));
    end
    image = Rawimage / max(Rawimage(:));
    
    % spatial frequency
    F_image = zeros(size(image));
    for i = 1:Nraw
        F_image(:,:,i) = fftshift(fft2(ifftshift(image(:,:,i))));
    end
    
    % choose two spatial frequencies of illumination pattern to 
    % estimate the phase shifts, they can be acquired from the Fourier spectrum of 
    % the raw images (data tip of matlab tool)
    SpatialFrequency = [-7.413e-4 -3.3213;1.9177 -0.0018];
    period = [1/abs(SpatialFrequency(1,2)) 1/abs(SpatialFrequency(2,1))];
    
    % auto-correlation method. Referrence 
    % A. Lal, C. Shan, and P. Xi, "Structured Illumination Microscopy Image Reconstruction Algorithm," 
    % IEEE J. Sel. Top. Quantum Electron. 22, 50¨C63 (2016).
    phase = zeros(Nraw,2);
    for m = 1:Nraw
        Fcc = F_image(:,:,m).*conj(OTF);   
        cc = fftshift(ifft2(ifftshift(Fcc)));
        Fcc_move1 = fftshift(fft2(ifftshift(cc.*exp(1i*2*pi...
            *(SpatialFrequency(1,1).*x+SpatialFrequency(1,2).*y)))));
        Fcc_move2 = fftshift(fft2(ifftshift(cc.*exp(1i*2*pi...
            *(SpatialFrequency(2,1).*x+SpatialFrequency(2,2).*y)))));
        fre_value1 = sum(sum(Fcc.*conj(Fcc_move1)));
        fre_value2 = sum(sum(Fcc.*conj(Fcc_move2)));
        phase(m,1) = angle(fre_value1);
        phase(m,2) = angle(fre_value2);
    end
    phase = phase-repmat(phase(1,:),Nraw,1);

    % phase unwrap
    tmp = zeros(size(phase));
    tmp(:,1) = unwrap(phase(:,2)); tmp(:,2) = unwrap(-1.*phase(:,1));
    
    % change phase to pratical position
    position(:,1) = tmp(:,1)*period(2)/2/pi;
    position(:,2) = tmp(:,2)*period(1)/2/pi;

end
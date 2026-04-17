function System = Get_system_parameter(Nx,Ny)
    % set parameter           
    System.NA = 1.45;             % numerical aperture
    System.lambda_illu = 0.488;	% illumination wavelength/um
    System.lambda_ex=0.519;     % excitation wavelength/um

    % set coordinate
    System.dx = 6.5/100/1.5;    % pixel size/mag1/mag2
    dx = System.dx;
    Lfx = 1/dx; Lfy = 1/dx;
    n = (1:Nx)'; m = (1:Ny)';
    x = (-Nx/2:1:Nx/2-1)*dx;
    y = (-Ny/2:1:Ny/2-1)*dx;                                          %spatial coordinate
    [x,y] = meshgrid(x,y); fx_nom = -Lfx/2+Lfx/(Nx)*(n-1);               %spatial frequency coordinate
    fy_nom = -Lfy/2+Lfy/(Ny)*(m-1);
    [fx,fy] = meshgrid(fx_nom,fy_nom);
    System.x = x; System.y = y; System.fx = fx; System.fy = fy;
    

    rho0 = System.NA/System.lambda_ex;                              % cut-off frequency
    rho = ((fx).^2 + (fy).^2).^0.5;


    %load experimental psf
    psf = zeros(Ny,Nx);
    pp = load('psf.mat');
    psfdata = pp.psf_ave;
    [Height,Width] = size(psfdata);
    tmpNx = ceil((Nx-Width)/2);tmpNy = ceil((Ny-Height)/2);
    psf(tmpNy+1:tmpNy+Height,tmpNx+1:tmpNx+Width) = psfdata;
    
    OTF = (fftshift(fft2(ifftshift(psf))));
    OTF = abs(OTF)/max(max(abs(OTF)));
    OTF = OTF.*(rho<=2*rho0);
    System.OTF = OTF; System.psf = psf;
end
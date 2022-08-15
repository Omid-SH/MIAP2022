% ANISODIFF - Anisotropic diffusion.
%
% 
%  diff = anisodiff(im, niter, kappa, lambda, option)
%
% 
%         im     - input image
%         niter  - number of iterations.
%         kappa  - conduction coefficient 20-100
%         lambda - max value of .25 for stability
%         option - 1 Perona Malik diffusion equation No 1
%                  2 Perona Malik diffusion equation No 2
%
% Return
%         diff   - diffused image.

% Reference: 
% P. Perona and J. Malik. 
% Scale-space and edge detection using anisotropic diffusion.
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 
% 12(7):629-639, July 1990.

function diff = anisodiff(im, niter, kappa, lambda, option)

if ndims(im)==3
  error('Anisodiff only operates on 2D grey-scale images');
end

im = double(im);
[rows,cols] = size(im);
diff = im;

for i = 1:niter

  diffl = zeros(rows+2, cols+2);
  diffl(2:rows+1, 2:cols+1) = diff;
  
  % North, South, East and West differences
  deltaN = diffl(1:rows,2:cols+1)   - diff;
  
  deltaS = diffl(3:rows+2,2:cols+1) - diff;
  
  deltaE = diffl(2:rows+1,3:cols+2) - diff;
  
  deltaW = diffl(2:rows+1,1:cols)   - diff;
  
  % Conduction
  if option == 1
    cN = exp(-(deltaN/kappa).^2);    
    cS = exp(-(deltaS/kappa).^2);
    cE = exp(-(deltaE/kappa).^2);
    cW = exp(-(deltaW/kappa).^2);    
  elseif option == 2
    cN = 1./(1 + (deltaN/kappa).^2);
    cS = 1./(1 + (deltaS/kappa).^2);
    cE = 1./(1 + (deltaE/kappa).^2);
    cW = 1./(1 + (deltaW/kappa).^2);
  end
  % APPLYING FOUR-POINT-TEMPLETE FOR numerical solution of DIFFUSION P.D.E.
  diff = diff + lambda*(cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW);

end
% fprintf('\n');
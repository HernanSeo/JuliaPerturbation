%This program evaluates the analytical first and second (if approx=2) derivatives of f numerically. The parameters and steady state values of the arguments of the function f are assumed to be in the workspace. Also, the order of approximation must be in the workspace.
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001
%Changed on September 25, 2001 to replace subs with eval and make it no longer a function.


nfx = zeros(size(fx));
nfx(:) = eval(fx(:));

nfxp = zeros(size(fxp));
nfxp(:)= eval(fxp(:));

nfy = zeros(size(fy));
nfy(:) = eval(fy(:));

nfyp = zeros(size(fyp));
nfyp(:)= eval(fyp(:));

nf = zeros(size(f));
nf(:)=eval(f(:));

if approx==1
   
%If only a first-order approximation is desired, set all second derivatives equal to zero
nfypyp=0; nfypy=0; nfypxp=0; nfypx=0; nfyyp=0; nfyy=0; nfyxp=0; nfyx=0; nfxpyp=0; nfxpy=0; nfxpxp=0; nfxpx=0; nfxyp=0; nfxy=0; nfxxp=0; nfxx=0;
   
   else

nfypyp=zeros(size(fypyp));
nfypyp(:)=eval(fypyp(:));

nfypy=zeros(size(fypy));
nfypy(:)=eval(fypy(:));

nfypxp=zeros(size(fypxp));
nfypxp(:)=eval(fypxp(:));

nfypx=zeros(size(fypx));
nfypx(:)=eval(fypx(:));

nfyyp=zeros(size(fyyp));
nfyyp(:)=eval(fyyp(:));

nfyy=zeros(size(fyy));
nfyy(:)=eval(fyy(:));

nfyxp=zeros(size(fyxp));
nfyxp(:)=eval(fyxp(:));

nfyx=zeros(size(fyx));
nfyx(:)=eval(fyx(:));

nfxpyp=zeros(size(fxpyp));
nfxpyp(:)=eval(fxpyp(:));

nfxpy=zeros(size(fxpy));
nfxpy(:)=eval(fxpy(:));

nfxpxp=zeros(size(fxpxp));
nfxpxp(:)=eval(fxpxp(:));

nfxpx=zeros(size(fxpx));
nfxpx(:)=eval(fxpx(:));

nfxyp=zeros(size(fxyp));
nfxyp(:)=eval(fxyp(:));

nfxy=zeros(size(fxy));
nfxy(:)=eval(fxy(:));

nfxxp=zeros(size(fxxp));
nfxxp(:)=eval(fxxp(:));

nfxx=zeros(size(fxx));
nfxx(:)=eval(fxx(:));


end 
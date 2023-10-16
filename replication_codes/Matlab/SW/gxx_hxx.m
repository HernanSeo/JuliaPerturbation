%GXX_HXX.M
%[gxx,hxx] = gxx_hxx(fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,hx,gx) 
%finds the 3-dimensional arrays gxx and hxx necessary to compute the 2nd order approximation 
%to the decision rules of a DSGE model of the form E_tf(yp,y,xp,x)=0, with solution 
%xp=h(x,sigma) + sigma * eta * ep and y=g(x,sigma). For more details, see  
%``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy 
%Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, JEDC, January 2004, p. 755-775. 
%
%INPUTS: First and second derivatives of f and first-order approximation to the functions g and 
%h: fx, fxp, fy, fyp, fypyp, fypy, fypxp, fypx, fyyp, fyy, fyxp, fyx, fxpyp, fxpy, fxpxp, fxpx, 
%fxyp, fxy, fxxp, fxx, hx, gx
%
%OUTPUTS: Second-order derivatives of the functions g and h with respect to x, evaluated 
%at (x,sigma)=(xbar,0), where xbar=h(xbar,0). That is, hxx gxx
%
% We solve a linear system of the type q = Q * x where x is a vector containing the elements of 
%gxx and hxx appropritely stacked and q and Q are, respectively, a vector and a matrix whose 
%elements are functions of the inputs of the program. 
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%
%Date February 18, 2004

function [gxx,hxx] = gxx_hxx(fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,hx,gx)

m=0;
nx = size(hx,1); %rows of hx and hxx
ny = size(gx,1); %rows of gx and gxx
n = nx + ny; %length of f
ngxx = nx^2*ny; %elements of gxx

sg = [ny nx nx]; %size of gxx
sh = [nx nx nx]; %size of hxx

Q = zeros(n*nx*(nx+1)/2,n*nx*nx);
q = zeros(n*nx*(nx+1)/2,1);
gxx=zeros(sg);
hxx=zeros(sh);
GXX=zeros(sg);
HXX=zeros(sh);


for i=1:n
for j=1:nx
%for k=1:nx
for k=1:j
m = m+1;

%First Term
q(m,1) = ( shiftdim(fypyp(i,:,:),1) * gx * hx(:,k) + shiftdim(fypy(i,:,:),1) * gx(:,k) + shiftdim(fypxp(i,:,:),1) *  hx(:,k) + shiftdim(fypx(i,:,k),1) )' * gx * hx(:,j); 

% Second term

GXX(:) = kron(ones(nx^2,1),fyp(i,:)');

pGXX = permute(GXX,[2 3 1]);
pGXX(:) = pGXX(:) .* kron(ones(nx*ny,1),hx(:,j));
GXX=ipermute(pGXX,[2 3 1]);

pGXX = permute(GXX,[3 1 2]);
pGXX(:) = pGXX(:) .* kron(ones(nx*ny,1),hx(:,k));
GXX=ipermute(pGXX,[3 1 2]);

Q(m,1:ngxx)=GXX(:)';

GXX=0*GXX;

%Third term

HXX(:,j,k) = (fyp(i,:) * gx)';

Q(m,ngxx+1:end)=HXX(:)';

HXX = 0*HXX;

%Fourth Term
q(m,1) = q(m,1) + ( shiftdim(fyyp(i,:,:),1) * gx * hx(:,k) +  shiftdim(fyy(i,:,:),1) * gx(:,k) + shiftdim(fyxp(i,:,:),1) * hx(:,k) +  shiftdim(fyx(i,:,k),1) )' * gx(:,j); 

%Fifth Term

GXX(:,j,k)=fy(i,:)';

Q(m,1:ngxx) = Q(m,1:ngxx) + GXX(:)';

GXX = 0*GXX;

%Sixth term
q(m,1) = q(m,1) + ( shiftdim(fxpyp(i,:,:),1) * gx * hx(:,k) + shiftdim(fxpy(i,:,:),1) * gx(:,k) + shiftdim(fxpxp(i,:,:),1) * hx(:,k) + fxpx(i,:,k)')' * hx(:,j);


%Seventh Term

HXX(:,j,k)=fxp(i,:)';

Q(m,ngxx+1:end) = Q(m,ngxx+1:end) + HXX(:)';

HXX = 0*HXX;

%Eighth Term
q(m,1) = q(m,1) +  shiftdim(fxyp(i,j,:),1) * gx * hx(:,k) +  shiftdim(fxy(i,j,:),1) * gx(:,k) +  shiftdim(fxxp(i,j,:),1) * hx(:,k) + fxx(i,j,k);

end %k 
end %j
end %i

A = temp(nx,ny);

Qt = Q* A;

xt = -Qt\q;
x = A* xt;

gxx(:)=x(1:ngxx);
hxx(:) = x(ngxx+1:end);



function A = temp(nx,ny) %subfunction
%function A = temp(nx,ny)
%This function creates a matrix A, size n*nx^2 by n*nx*(nx+1)/2, such that x = A xtilde, 
%where x is the vector  [gxx(:); hxx(:)] and xtilde is a vector containing
%all the elements  gxx(i,j,k) and hxx(i,j,k), respectively, such that j<=
%k, that is, it is a subset of the elements in the vector x that appear only once.
%This is so because gxx(i,j,k) is symmetric with respect to j and k, that is, gxx(i,j,k)=gxx(i,k,j).
%The reason we use this program is that the matrix we invert (which in the earlier versioni of this program used to be
%Q and now is Qt) is of  size n*nx*(nx+1)/2 by n*nx*(nx+1)/2 rather than of size
%n*nx^2 by n*nx^2. This reduces computation time by about 30 percent. (Feb 16, 2004)

Ahxx=zeros(nx^3, nx^2*(nx+1)/2);
Agxx=zeros(ny*nx^2, ny*nx*(nx+1)/2);
mx=0;
my=0;
for k=1:nx;
    for j=k:nx;
        for i=1:nx; 
            mx=mx+1;
            Ahxx((j-1)*nx+i+(k-1)*nx*nx, mx) = 1;
            Ahxx((k-1)*nx+i+(j-1)*nx*nx, mx) = 1;
        end %i=1:nx
        for i=1:ny; 
            my=my+1;
            Agxx((j-1)*ny+i+(k-1)*ny*nx, my) = 1;
            Agxx((k-1)*ny+i+(j-1)*ny*nx, my) = 1;
        end %i=1:ny
    end %j
end %k

A = zeros((nx+ny)*nx^2,(nx+ny)*nx*(nx+1)/2);
A(1:ny*nx^2, 1:ny*nx*(nx+1)/2)=Agxx;
A(ny*nx^2+1:end, ny*nx*(nx+1)/2+1:end)=Ahxx;
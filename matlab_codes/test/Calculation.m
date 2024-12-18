tic
global data;
global k;
global qq;
global distances;

k=2;
%p=10;
data=xlsread('C:\Users\Apollo\Downloads\Hexagons.xlsx','B2:U1813');
%data=generateData15d(100000,1);
%newData=zeros(10000,3);
%for i=1:10000
 %   phi=acos(2*rand(1)-1);
 %   tetta=2*pi*rand(1);
 %   x=p*sin(phi)*cos(tetta);
 %   y=p*sin(phi)*sin(tetta);
 %   z=p*cos(phi);
 %   newData(i,1)=x;
 %   newData(i,2)=y;
 %   newData(i,3)=z;
   %disp(i);
%end
%data=newData;
distances=calculateDistancesBetweenAllPoints(data);
max_distance=max(distances(:));
distances=distances/max_distance;
scatter3(data(:,1),data(:,2),data(:,3));
results=zeros(3,35);
dobavka=0;
for iter=1:35
qq=-1+0.1*iter+dobavka;
if qq==1
    dobavka=dobavka+0.1;
    qq=qq+dobavka;
end
betta0=1;
options = optimset('MaxFunEvals',1200, 'MaxIter', 800, 'Display', 'iter');
[bettas,Fval]=fsolve(@qFunc,betta0,options);
disp('q:')
disp(qq);
disp(bettas);
disp(Fval);
results(1,iter)=qq;
results(2,iter)=bettas/(1-qq);
results(3,iter)=Fval;
end
disp('--------------');
for i=1:length(results(1,:))
    disp('q=');
    disp(results(1,i));
    disp('betta=');
    disp(results(2,i));
    disp('Fval=');
    disp(results(3,i));
end
p1=polyfit(results(1,:),results(2,:),0);
disp(p1);
xx = linspace(results(1,1), results(1,end), 200);
yy = polyval(p1, xx);
plot(results(1,:),results(2,:),'o',xx,yy);
legend('DATA', '{\itp}^{(0)}({\itx})');

toc
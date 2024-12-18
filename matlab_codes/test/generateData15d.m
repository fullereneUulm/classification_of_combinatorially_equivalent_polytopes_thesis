function [newData] = generateData15d(numberOfExamples,coefficientBoundary)
eps=0.1;
numOfEquations=15;
numOfCoefficients=20;
numOfVariables=20;
B=rand(numOfEquations,numOfCoefficients)*(2.*coefficientBoundary)-coefficientBoundary;
C=ones(numOfEquations,1);
newData=zeros(numberOfExamples,numOfVariables);
%------------TEST-----------
Z=null(B);
p1=B\C;

for i=1:numberOfExamples
    temp=rand(numOfVariables-numOfEquations,1)*12-6;
    eps1=rand(numOfVariables,1)*2*eps-eps
    vect=p1+eps1+Z*temp;
    newData(i,:)=vect';
end
csvwrite('data.csv', newData)
%scatter3(newData(:,1),newData(:,2),newData(:,3),'.')
end

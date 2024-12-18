function [ result ] = rankMinimum(j)

global data;
global k;
global distances;
distances_temp(1:(length(distances)-1))=0;
counter=1;
for i=1:length(data)
    if (i~=j) 
        distances_temp(counter)=distances(i,j);
        counter=counter+1;
    end
end
distances_temp=sort(distances_temp);
result=distances_temp(k);
end



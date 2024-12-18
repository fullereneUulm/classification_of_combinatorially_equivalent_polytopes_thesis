function [distances] = calculateDistancesBetweenAllPoints(points)
distances=zeros(length(points));
for i=1:length(distances)
    for j=1:length(distances(1,:))
        distances(i,j)=EuclideanDistance(points(i,:),points(j,:));
    end
end

end


function p = plot_isosurface(ax,x,y,z,V,isovalue,colors)
    [X,Y,Z] = meshgrid(x,y,z);
    shape(cell2mat(X))
    shape(cell2mat(Y)) 
    shape(cell2mat(Z))
    shape(cell2mat(V))
    p = patch(ax,isosurface(X,Y,Z,V,isovalue,colors));
    set(p,'FaceColor','interp','EdgeColor','none');
    colormap(hot(8));
    hold(ax,'on');
end
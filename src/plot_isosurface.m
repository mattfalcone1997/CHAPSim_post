function p = plot_isosurface(ax,isovalue,color)
    load('.temp.mat','x','y','z','V')
    [X,Y,Z] = meshgrid(x,y,z);
    % size(X)
    % size(Y)
    % size(Z)
    % size(V)
    p = patch(ax,isosurface(Y,Z,X,V,isovalue,color));
    set(p,'FaceColor',color,'EdgeColor','none');
    view(ax,37.5,30)
    % rotate(p,[1,0,0],90);
    axis(ax,'tight')
    pbaspect(ax,[max(y)-min(y),max(z)-min(z),max(x)-min(x)])
    hold(ax,'on');
    % camlight
    lighting gouraud
end
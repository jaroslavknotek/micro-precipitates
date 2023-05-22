function refineMask()
% expects ImgName folder containing mask.png and img.png
% returns user-refined mask2.png

%% Global variables
InROI=false;
TheID=nan;
Points=cell(1);
ImgName='D:\Git_Repos\TrainingData\Train_05-22\X\012_SMMAG_x300k_613';

%% Load and process images
Img=imread(fullfile(ImgName,'img.png'));
Mask=imread(fullfile(ImgName,'mask.png'));
% Get outlines
Outlines=bwmorph(imfill(Mask),'remove');
% Label features
Mask=bwlabel(Mask);
Outlines=Outlines.*Mask;

%% Plot image
imagesc(Img,'ButtonDownFcn',@(s,e)imClick(s,e))
colormap('bone')

% define empty ROI plot
ROI=drawpolygon(gca,'Visible','off','Position',[1,1;2,2]);

% define empty outline plot
Outline(max(Mask,[],'all'))=line(nan,nan);

%% Process outlines and assign polygons
for ii=1:max(Mask,[],'all') % 186
    % find ii-th outline
    [y,x]=find(Outlines==ii);
    % sort point cluster into polygon
    [xy]=sortPoints([x,y]);
%     close all
    % remove redundant points using Ramer-Douglas-Peucker algorithm
    xy=RDP(xy,.1);
    % update line object and Points matrix
    Outline(ii)=line([xy(:,1);xy(1,1)],[xy(:,2);xy(1,2)],'marker','.','color','r',...
        'buttondownfcn',@(s,e)lClick(s,e),'UserData',ii);
    Points{ii}=xy;

end


%% Nested functions
    % sort cluster points into a polygon
    function[sXY]=sortPoints(uXY)
        % find the closest point to last polygon point in the cluster, pop it from cluster and
        % append it polygon.
        sXY=nan(size(uXY));
        CI=1;
        XY=uXY;
        for si=1:size(uXY,1)
            sXY(si,:)=XY(CI,:);
            XY=XY([1:CI-1,1+CI:end],:);
            EuD=sum([XY(:,1)-sXY(si,1),XY(:,2)-sXY(si,2)].^2,2);
            [~,CI]=min(EuD);
        end
    end

    % Ramer - Douglas - Peucker algorithm
    function[hull]=RDP(hull,eps)
        % Find the farthest point X from first-last line.
        % If the distance is below [eps], return first and last point
        % Otherwise split the line in first-X and X-last branches and apply
        % RDP recursively.

        sp=hull(1,:);
        ep=hull(end,:);
        ip=hull(2:end-1,:);
        % calculate distances of inner points from the first-last line
        dst=PerpDist(ip,sp,ep);
        % find the point farthest from the f-l line
        [mx,mi]=max(dst);
        if mx>eps % farthest point does not fit in.
            lp=[sp;ip(1:mi,:)];
            if size(lp,1)>2
                lp=RDP(lp,eps);
            end
            rp=[ip(mi:end,:);ep];
            if size(rp,1)>2
                rp=RDP(rp,eps);
            end
            hull=[lp;rp(2:end,:)];
        else
            hull=[sp;ep];
        end
    end

    % Perpendicular distance of a point to a line
    function[D]=PerpDist(PDX,varargin)
        % PDX - point of interest
        % varargin - line-defining points
        % PerpDist(A,B) return euclidean distance between A and B
        % PerpDist(A,B,C) return distance between A and |-BC-|

        D=nan(size(PDX,1),1);
        if nargin<2+1 % edge is defined by one point, use euclidean distance between points
            PDA=varargin{1};
            for PDi=1:size(PDX,1)
                D(PDi)=norm(PDX(PDi,:)-PDA);
            end
        else % edge is a line, use eucleidean distance from a line
            PDA=varargin{1};
            PDB=varargin{2};
            if PDA==PDB
                D=PerpDist(PDX,PDA);
            else
                for PDi=1:size(PDX,1)
                    D(PDi)=abs(PDA(1)*(PDB(2)-PDX(PDi,2))+PDB(1)*(PDX(PDi,2)-PDA(2))+PDX(PDi,1)*(PDA(2)-PDB(2)))/norm(PDB-PDA);
                end
            end
        end
    end
    
    % update the Mask array and mask2.png file
    function updateMask()
        % delete old mask data for TheID-th feature
        Mask(Mask==TheID)=0;
        % create and label mask for updated TheID-th feature
        Layer=poly2mask(Points{TheID}(:,1),Points{TheID}(:,2),size(Mask,1),size(Mask,2))*TheID;
        % update line data
        set(Outline(TheID),'xdata',[Points{TheID}(:,1);Points{TheID}(1,1)],'ydata',[Points{TheID}(:,2);Points{TheID}(1,2)]);
        
        % update mask array
        Mask=Mask+Layer;
        % update mask2.png image
        imwrite(Mask>0,fullfile(ImgName,'mask2.png'));
    end
    
    % image click callback
    function imClick(~,e)
        % when editing polygon, respond only to right click (button 3) -
        % deactivate RIO and update masks
        % otherwise check Mask what feature is to be selected.
        % If nonzero, activate ROI and deactivate Outlines.
%         InROI,e
        if InROI
            if e.Button==3
                InROI=false;
                set(Outline,'visible','on')
                Points{TheID}=ROI.Position;
                ROI.Visible='off';
                updateMask();
            end
        else
            Click=floor(e.IntersectionPoint);
            X=Click(1);
            Y=Click(2);
    
            ID=Mask(Y,X);
            if ID~=0
                InROI=true;
                set(Outline,'visible','off');
                set(Outline(ID),'visible','on');
                TheID=ID;
                ROI.Visible='on';
%                 ID
                size(Points{ID})
                ROI.Position=Points{ID};
            else
                
            end
        end

    end

    % Outline callback
    function lClick(s,e)
    % if the line is selected by left-click (button 1) activate ROI and
    % deactivate outlines.
    % if selected by right-click (button 3) delete the line and erase from
    % masks. Update masks.
        switch e.Button
            case 1
                InROI=true;
                TheID=s.UserData;
                ROI.Visible='on';
                ROI.Position=Points{TheID};
                set(Outline,'visible','off');
                set(Outline(TheID),'visible','off');
            case 3
                set(Outline(s.UserData),'xdata',nan,'ydata',nan);
                Mask(Mask==s.UserData)=0;
                Points{s.UserData}=nan(1,2);
                updateMasks()
        end
    end
end
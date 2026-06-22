% Generate ground-truth reference data for the NSDFB / NSCT, to validate the
% Julia port against. Run from the matlab/ folder.
%   matlab -batch "gen_reference"
addpath(genpath(fullfile(pwd, 'Nonsubsampled-Contourlet-Toolbox', 'nsct_toolbox')));

here = fileparts(mfilename('fullpath'));
nsctdir = fullfile(here, 'Nonsubsampled-Contourlet-Toolbox', 'nsct_toolbox');
cd(nsctdir);

% Build MEX if needed
if exist(['atrousc.' mexext], 'file') ~= 3
    mex atrousc.c;
end
if exist(['zconv2.' mexext], 'file') ~= 3
    mex zconv2.c;
end
if exist(['zconv2S.' mexext], 'file') ~= 3
    mex zconv2S.c;
end

outdir = fullfile(here, '.ref');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Deterministic input
n = 32;
[ii, jj] = ndgrid(1:n, 1:n);
x = sin(2*pi*0.1*ii) + cos(2*pi*0.07*jj) + 0.5*sin(2*pi*0.05*(ii+jj));
x = double(x);

dfilt = 'pkva';   % matches Julia Q2345

% Dump the base 2-D filters so Julia can validate filter construction
[h1d, h2d] = dfilters(dfilt, 'd');  h1d = h1d./sqrt(2); h2d = h2d./sqrt(2);
[h1r, h2r] = dfilters(dfilt, 'r');  h1r = h1r./sqrt(2); h2r = h2r./sqrt(2);
k1 = modulate2(h1d, 'c'); k2 = modulate2(h2d, 'c');
[f1, f2] = parafilters(h1d, h2d);
writematrix(h1d, fullfile(outdir, 'h1d.csv'));
writematrix(h2d, fullfile(outdir, 'h2d.csv'));
writematrix(k1,  fullfile(outdir, 'k1.csv'));
writematrix(k2,  fullfile(outdir, 'k2.csv'));
for i = 1:4
    writematrix(f1{i}, fullfile(outdir, sprintf('f1_%d.csv', i)));
    writematrix(f2{i}, fullfile(outdir, sprintf('f2_%d.csv', i)));
end

writematrix(x, fullfile(outdir, 'x.csv'));

% NSDFB decompositions at several levels
for L = 1:4
    y = nsdfbdec(x, dfilt, L);
    for k = 1:numel(y)
        writematrix(y{k}, fullfile(outdir, sprintf('nsdfb_L%d_sb%02d.csv', L, k)));
    end
    rec = nsdfbrec(y, dfilt);
    writematrix(rec, fullfile(outdir, sprintf('nsdfb_L%d_rec.csv', L)));
    fprintf('L=%d: %d subbands, PR err = %.3e\n', L, numel(y), max(abs(rec(:)-x(:))));
end

fprintf('Reference data written to %s\n', outdir);

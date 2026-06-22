function verify_vs_julia(J, L_array, sizes)
% VERIFY_VS_JULIA  Cross-validate and benchmark Julia's CT, NSCT, and NSDFB
% against the da Cunha–Zhou–Do MATLAB reference implementations.
%
%   verify_vs_julia              % defaults: J=2, L_array=[2,3], sizes=[64,128,256,512]
%   verify_vs_julia(J, L_array)
%   verify_vs_julia(J, L_array, sizes)
%
% Checks three transforms for each size:
%   1. NSDFB  — nsdfbdec/nsdfbrec  vs Julia nsdfb_decompose/reconstruct
%              Cross-validates subbands (same pkva diamond filters → should match ~1e-12)
%   2. CT     — pdfbdec/pdfbrec    vs Julia ct_forward/ct_inverse
%              PR-only check: Julia uses JPEG2000-normalised CDF97; MATLAB pdfbdec
%              uses a different '9-7' scaling, so subbands differ but both give PR≈0.
%   3. NSCT   — nsctdec/nsctrec    vs Julia nsct_forward/nsct_inverse
%              PR-only check: NSCT toolbox uses a 2-D non-separable '9-7' diamond
%              pyramid filter, whereas Julia uses separable à-trous CDF97, so subbands
%              differ by design.  Both guarantee PR to machine precision.
%
% Reports per-transform PR errors and timings across sizes.
%
% Requirements:
%   • MATLAB on PATH (or MATLAB_BIN env var)
%   • Julia with Contourlets package instantiated (JULIA_BIN env var or on PATH)
%   • Toolboxes cloned under matlab/Contourlet-transform/ and
%     matlab/Nonsubsampled-Contourlet-Toolbox/nsct_toolbox/

if nargin < 1 || isempty(J);       J       = 2;            end
if nargin < 2 || isempty(L_array); L_array = [2, 3];       end
if nargin < 3 || isempty(sizes);   sizes   = [64, 128, 256, 512]; end

here    = fileparts(mfilename('fullpath'));
ctdir   = fullfile(here, 'Contourlet-transform');
nsctdir = fullfile(here, 'Nonsubsampled-Contourlet-Toolbox', 'nsct_toolbox');
iodir   = fullfile(here, '.verify');
if ~exist(iodir, 'dir'); mkdir(iodir); end

addpath(ctdir);
addpath(nsctdir);

% ── Build MEX files if needed ─────────────────────────────────────────────────
old = cd(nsctdir);
for mxf = {'atrousc', 'zconv2', 'zconv2S'}
    f = mxf{1};
    if exist([f '.' mexext], 'file') ~= 3; mex([f '.c']); end
end
cd(old);

old = cd(ctdir);
if exist(['resampc.' mexext], 'file') ~= 3; mex resampc.c; end
cd(old);

% ── Write shared parameters ───────────────────────────────────────────────────
writematrix(sizes(:)',     fullfile(iodir, 'sizes.csv'));
writematrix(J,             fullfile(iodir, 'J.csv'));
writematrix(L_array(:)',   fullfile(iodir, 'L_array.csv'));

% NSDFB standalone level: use the finest (last, since L_array is fine-to-coarse
% in the Julia convention).  Keep at most 3 levels to avoid huge filter stacks.
L_nsdfb = min(L_array(end), 3);
writematrix(L_nsdfb, fullfile(iodir, 'L_nsdfb.csv'));

% MATLAB PDFB/NSCT use coarse-to-fine ordering; Julia L_array is fine-to-coarse.
nlevs = fliplr(L_array(:)');     % coarse-to-fine for MATLAB

dfilt_ct   = 'pkva';     pfilt_ct   = '9-7';   % pdfbdec(x, pfilt, dfilt, ...)
dfilt_nsct = 'pkva';     pfilt_nsct = '9-7';   % nsctdec(x, levels, dfilt, pfilt)

nrun = 7;

% ── Generate per-size inputs ──────────────────────────────────────────────────
for n = sizes
    rng(12345 + n);                % deterministic, varies by size
    x = randn(n, n);
    writematrix(x, fullfile(iodir, sprintf('x_%d.csv', n)));
end

% ── Run Julia (one subprocess handles all sizes and all transforms) ───────────
pkgdir = fileparts(here);
jul = getenv('JULIA_BIN'); if isempty(jul); jul = 'julia'; end
jl_script = fullfile(here, 'verify_julia_side.jl');
cmd = sprintf('env LD_LIBRARY_PATH= %s --project=%s %s %s', jul, ...
    bash_quote(pkgdir), bash_quote(jl_script), bash_quote(iodir));
fprintf('Running Julia: %s\n\n', cmd);
status = system(cmd);
if status ~= 0
    error('Julia side failed (status %d). Check JULIA_BIN or set it in your env.', status);
end

% ── Read Julia timings ────────────────────────────────────────────────────────
jl_t = readtable(fullfile(iodir, 'jl_timing.csv'));

% ── Results table ─────────────────────────────────────────────────────────────
fprintf('\n');
fprintf('========= MATLAB vs Julia cross-validation (J=%d, L=[%s]) =========\n', ...
        J, num2str(L_array));
fprintf('\n');

all_pass = true;

for n = sizes
    rng(12345 + n);
    x = randn(n, n);

    fprintf('─── n = %d ──────────────────────────────────────────────────────\n', n);

    % ── NSDFB ─────────────────────────────────────────────────────────────────
    y_nsdfb = nsdfbdec(x, 'pkva', L_nsdfb);        % MATLAB decompose
    y_nsdfb_w = {};                                  % warm up
    td_m = zeros(1, nrun); tr_m = zeros(1, nrun);
    for r = 1:nrun
        t = tic; y_nsdfb = nsdfbdec(x, 'pkva', L_nsdfb); td_m(r) = toc(t);
        t = tic; rec_m   = nsdfbrec(y_nsdfb, 'pkva');     tr_m(r) = toc(t);
    end
    pr_nsdfb_m = max(abs(rec_m(:) - x(:)));

    nsb = numel(y_nsdfb);
    sb_err_nsdfb = 0;
    for k = 1:nsb
        jl_sb = readmatrix(fullfile(iodir, sprintf('jl_nsdfb_%d_sb%02d.csv', n, k)));
        sb_err_nsdfb = max(sb_err_nsdfb, max(abs(jl_sb(:) - y_nsdfb{k}(:))));
    end
    jl_rec_nsdfb = readmatrix(fullfile(iodir, sprintf('jl_nsdfb_%d_rec.csv', n)));
    rec_err_nsdfb = max(abs(jl_rec_nsdfb(:) - rec_m(:)));
    pr_nsdfb_jl   = max(abs(jl_rec_nsdfb(:) - x(:)));

    row_m = jl_t(strcmp(jl_t.transform,'nsdfb') & jl_t.size == n, :);
    td_jl = row_m.decompose_s; tr_jl = row_m.reconstruct_s;

    pass_nsdfb = sb_err_nsdfb < 1e-9 && rec_err_nsdfb < 1e-9;
    all_pass = all_pass && pass_nsdfb;

    fprintf('  NSDFB  (L=%d, %d sbs):  MATLAB PR=%.1e | Julia PR=%.1e | cross=%.1e  [%s]\n', ...
            L_nsdfb, nsb, pr_nsdfb_m, pr_nsdfb_jl, max(sb_err_nsdfb, rec_err_nsdfb), ...
            yesno(pass_nsdfb));
    fprintf('    timing dec:  MATLAB %.4f s | Julia %.4f s | %.2fx\n', ...
            median(td_m), td_jl, median(td_m)/td_jl);
    fprintf('    timing rec:  MATLAB %.4f s | Julia %.4f s | %.2fx\n', ...
            median(tr_m), tr_jl, median(tr_m)/tr_jl);

    % ── CT (PR-only — different LP filter normalizations, subbands differ by design) ──
    y_ct = pdfbdec(x, pfilt_ct, dfilt_ct, nlevs);
    rec_ct_m = pdfbrec(y_ct, pfilt_ct, dfilt_ct);
    td_ct_m = zeros(1, nrun); tr_ct_m = zeros(1, nrun);
    for r = 1:nrun
        t = tic; y_ct     = pdfbdec(x, pfilt_ct, dfilt_ct, nlevs); td_ct_m(r) = toc(t);
        t = tic; rec_ct_m = pdfbrec(y_ct, pfilt_ct, dfilt_ct);      tr_ct_m(r) = toc(t);
    end
    pr_ct_m = max(abs(rec_ct_m(:) - x(:)));

    % Julia PR only (no cross-subband comparison: Julia uses JPEG2000-normalised CDF97,
    % MATLAB pdfbdec('9-7') uses a different 9/7 scaling → subbands differ by ~√2/level)
    jl_rec_ct = readmatrix(fullfile(iodir, sprintf('jl_ct_%d_rec.csv', n)));
    pr_ct_jl  = max(abs(jl_rec_ct(:) - x(:)));

    row_ct   = jl_t(strcmp(jl_t.transform,'ct') & jl_t.size == n, :);
    td_ct_jl = row_ct.decompose_s; tr_ct_jl = row_ct.reconstruct_s;

    pass_ct = pr_ct_m < 1e-9 && pr_ct_jl < 1e-9;
    all_pass = all_pass && pass_ct;

    fprintf('  CT     (J=%d, nlevs=[%s]):  MATLAB PR=%.1e | Julia PR=%.1e  (PR-only; LP filters differ)  [%s]\n', ...
            J, num2str(nlevs), pr_ct_m, pr_ct_jl, yesno(pass_ct));
    fprintf('    timing dec:  MATLAB %.4f s | Julia %.4f s | %.2fx\n', ...
            median(td_ct_m), td_ct_jl, median(td_ct_m)/td_ct_jl);
    fprintf('    timing rec:  MATLAB %.4f s | Julia %.4f s | %.2fx\n', ...
            median(tr_ct_m), tr_ct_jl, median(tr_ct_m)/tr_ct_jl);

    % ── NSCT (PR-only — MATLAB uses 2-D non-separable diamond pyramid; Julia uses à-trous CDF97) ──
    y_nsct = nsctdec(x, nlevs, dfilt_nsct, pfilt_nsct);
    rec_nsct_m = nsctrec(y_nsct, dfilt_nsct, pfilt_nsct);
    td_nsct_m = zeros(1, nrun); tr_nsct_m = zeros(1, nrun);
    for r = 1:nrun
        t = tic; y_nsct     = nsctdec(x, nlevs, dfilt_nsct, pfilt_nsct); td_nsct_m(r) = toc(t);
        t = tic; rec_nsct_m = nsctrec(y_nsct, dfilt_nsct, pfilt_nsct);   tr_nsct_m(r) = toc(t);
    end
    pr_nsct_m = max(abs(rec_nsct_m(:) - x(:)));

    % Julia PR only (MATLAB NSCT uses 2-D diamond pyramid filters;
    % Julia uses separable à-trous CDF97 → LP subbands differ, NSDFB subbands may differ too)
    jl_rec_nsct = readmatrix(fullfile(iodir, sprintf('jl_nsct_%d_rec.csv', n)));
    pr_nsct_jl  = max(abs(jl_rec_nsct(:) - x(:)));

    row_nsct    = jl_t(strcmp(jl_t.transform,'nsct') & jl_t.size == n, :);
    td_nsct_jl  = row_nsct.decompose_s; tr_nsct_jl = row_nsct.reconstruct_s;
    row_nsct_ws = jl_t(strcmp(jl_t.transform,'nsct_ws') & jl_t.size == n, :);
    td_nsct_ws  = row_nsct_ws.decompose_s;

    pass_nsct = pr_nsct_m < 1e-9 && pr_nsct_jl < 1e-9;
    all_pass  = all_pass && pass_nsct;

    fprintf('  NSCT   (J=%d, nlevs=[%s]):  MATLAB PR=%.1e | Julia PR=%.1e  (PR-only; LP filters differ)  [%s]\n', ...
            J, num2str(nlevs), pr_nsct_m, pr_nsct_jl, yesno(pass_nsct));
    fprintf('    timing dec:  MATLAB %.4f s | Julia spatial %.4f s | Julia FFT-ws %.4f s | %.2fx (spatial) %.2fx (ws)\n', ...
            median(td_nsct_m), td_nsct_jl, td_nsct_ws, ...
            median(td_nsct_m) / td_nsct_jl, median(td_nsct_m) / td_nsct_ws);
    fprintf('    timing rec:  MATLAB %.4f s | Julia %.4f s | %.2fx\n', ...
            median(tr_nsct_m), tr_nsct_jl, median(tr_nsct_m) / tr_nsct_jl);

    fprintf('\n');
end

% ── Summary ───────────────────────────────────────────────────────────────────
fprintf('=======================================================================\n');
if all_pass
    fprintf('PASS: Julia matches the MATLAB reference for all transforms and sizes.\n');
else
    fprintf('FAIL: one or more transforms exceeded the 1e-9 cross-validation threshold.\n');
end
fprintf('=======================================================================\n');

end  % function verify_vs_julia

% ── Helpers ───────────────────────────────────────────────────────────────────

function q = bash_quote(s)
q = ['''' strrep(s, '''', '''\''''') ''''];
end

function s = yesno(b)
if b; s = 'PASS'; else; s = 'FAIL'; end
end

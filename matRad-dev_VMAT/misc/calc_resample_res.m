function [new_xres, new_yres, new_zres] = calc_resample_res( desired_xdims, desired_ydims, desired_zdims, xdims, ydims, zdims, x_res, y_res, z_res )

standard_ct_slice_dims = [512 512];

new_xres = xdims*x_res/desired_xdims;
new_yres = ydims*y_res/desired_ydims;
new_zres = zdims*z_res/desired_zdims;

end


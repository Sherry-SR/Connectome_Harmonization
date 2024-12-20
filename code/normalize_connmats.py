#! /usr/bin/env python

import sys 
import numpy as np
import nibabel as nib 

subjectlist, filename_template, gmwmi_template, output_template = sys.argv[1:]

subjects = [s.strip() for s in open(subjectlist,'r').readlines()]

filenames = [filename_template.format(s=s) for s in subjects]
gmwmifiles = [gmwmi_template.format(s=s) for s in subjects]

# gmwmis = [ nib.load(fn) for fn in gmwmifiles ] 
# connmats = [ np.loadtxt(fn) for fn in filenames ] 

def preprocess_connmat(connmat, gmwmi_img):
    zooms = gmwmi_img.header.get_zooms()
    voxels = ((gmwmi_img.get_fdata() > 0)*1).sum()
    volume = voxels*zooms[0]*zooms[1]*zooms[2]
    ret = (connmat + connmat.transpose())/2.0 
    np.fill_diagonal(ret, 0)
    ret = ret/volume 
    return ret 
    
for s, cf, gf in zip(subjects, filenames, gmwmifiles):
    c = np.loadtxt(cf)
    g = nib.load(gf)
    nc = preprocess_connmat(c, g)
    np.savetxt(output_template.format(s=s), nc, fmt="%0.8e")
    print('Wrote normalized connectome', output_template.format(s=s))
import SimpleITK as sitk

# Convert NIfTI to MHA
def convert_nifti_to_mha(input_file, output_file):
    image = sitk.ReadImage(input_file)
    sitk.WriteImage(image, output_file)

# Example usage
convert_nifti_to_mha("path/to/input.nii.gz", "path/to/output.mha")

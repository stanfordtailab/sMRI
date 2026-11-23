"""
reg_lib.py
Author : Mohammad H. Abbasi  (mabbasi [at] stanford.edu)  —  Stanford STAI Lab
License: MIT

register_image() lifted VERBATIM from sMRI/reg.py (lines 215-399).
The original reg.py runs a batch pipeline at module import time and cannot be
imported as a library, so the self-contained two-stage (rigid + affine)
registration function is extracted here unchanged for use by the Holoscan app.
Source: https://github.com/mhabbasiit/sMRI/blob/main/reg.py
"""
import os
import logging
import SimpleITK as sitk

logger = logging.getLogger("reg_lib")

# Referenced by register_image's outer except branch in the original module.
registration_results = {"failed_subjects": [], "errors": {}}


def register_image(fixed, moving, output_prefix, subject_id=None, base_name=None, other_dir=None):
    """
    Run a two-stage registration with SimpleITK:
    1. Rigid registration (rotation + translation)
    2. Affine registration (using rigid result as initial position)
    """
    try:
        output_dir = os.path.dirname(output_prefix)
        if not os.path.exists(output_dir):
            logger.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default values if not provided
        if base_name is None:
            base_name = os.path.basename(moving).replace("_brain.nii.gz", "")
        if other_dir is None:
            other_dir = os.path.join(output_dir, "other")
        os.makedirs(other_dir, exist_ok=True)
        
        logger.debug(f"Loading images for registration")
        # Load images using SimpleITK
        fixed_image = sitk.ReadImage(fixed, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving, sitk.sitkFloat32)
        
        logger.debug(f"Fixed image size: {fixed_image.GetSize()}, dimension: {fixed_image.GetDimension()}")
        logger.debug(f"Moving image size: {moving_image.GetSize()}, dimension: {moving_image.GetDimension()}")
        
        # Ensure both images are 3D
        if fixed_image.GetDimension() != 3 or moving_image.GetDimension() != 3:
            raise ValueError("Both images must be 3D")
            
        # STAGE 1: Rigid Registration
        logger.debug("Starting Rigid registration (Stage 1)")
        
        # Initialize registration method for Rigid
        rigid_registration = sitk.ImageRegistrationMethod()
        
        # Set up similarity metric for Rigid
        rigid_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        rigid_registration.SetMetricSamplingStrategy(rigid_registration.RANDOM)
        rigid_registration.SetMetricSamplingPercentage(0.25)
        # Note: SetMetricSamplingPercentageRandomSeed not available in SimpleITK 2.1.0
        
        # Set up interpolator for Rigid
        rigid_registration.SetInterpolator(sitk.sitkLinear)
        
        # Set up optimizer for Rigid
        rigid_registration.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                       numberOfIterations=1000,
                                                       convergenceMinimumValue=1e-6,
                                                       convergenceWindowSize=10)
        rigid_registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set up Rigid transform
        rigid_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
        rigid_registration.SetInitialTransform(rigid_transform)
        
        # Multi-resolution framework for Rigid
        rigid_registration.SetShrinkFactorsPerLevel([4, 2, 1])
        rigid_registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        
        try:
            # Execute Rigid registration
            final_rigid_transform = rigid_registration.Execute(fixed_image, moving_image)
            logger.debug("Rigid registration completed successfully")
            
            # Save Rigid transform for reference
            rigid_transform_path = os.path.join(other_dir, f"{base_name}_rigid.mat")
            sitk.WriteTransform(final_rigid_transform, rigid_transform_path)
            
            # Apply Rigid transform to get intermediate result
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(final_rigid_transform)
            
            rigid_moved_image = resampler.Execute(moving_image)
            
            # Save intermediate Rigid-registered image for reference
            rigid_warped_path = os.path.join(os.path.dirname(output_prefix), f"{base_name}_mni_rigid_warped.nii.gz")
            sitk.WriteImage(rigid_moved_image, rigid_warped_path)
            logger.debug(f"Saved Rigid-registered intermediate image to {rigid_warped_path}")
            
            # STAGE 2: Affine Registration (starting from Rigid result)
            logger.debug("Starting Affine registration (Stage 2) using Rigid result")
            
            # Initialize registration method for Affine
            affine_registration = sitk.ImageRegistrationMethod()
            
            # Set up similarity metric for Affine
            affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            affine_registration.SetMetricSamplingStrategy(affine_registration.RANDOM)
            affine_registration.SetMetricSamplingPercentage(0.25)
            # Note: SetMetricSamplingPercentageRandomSeed not available in SimpleITK 2.1.0
            
            # Set up interpolator for Affine
            affine_registration.SetInterpolator(sitk.sitkLinear)
            
            # Set up optimizer for Affine
            affine_registration.SetOptimizerAsGradientDescent(learningRate=0.05,  # Lower learning rate for fine-tuning
                                                            numberOfIterations=1000,
                                                            convergenceMinimumValue=1e-6,
                                                            convergenceWindowSize=10)
            affine_registration.SetOptimizerScalesFromPhysicalShift()
            
            # Initialize Affine transform from the Rigid result
            # Create a new Affine transform instead of trying to convert from Rigid
            affine_transform = sitk.AffineTransform(3)
            
            # Use identity matrix for initialization (standard starting point)
            # We'll use the Rigid-registered image as input, so we don't need 
            # to copy the rigid transformation parameters
            affine_registration.SetInitialTransform(affine_transform)
            
            # Multi-resolution framework for Affine (finer levels)
            affine_registration.SetShrinkFactorsPerLevel([2, 1])
            affine_registration.SetSmoothingSigmasPerLevel([1, 0])
            
            # Execute Affine registration
            final_affine_transform = affine_registration.Execute(fixed_image, rigid_moved_image)
            logger.debug("Affine registration completed successfully")
            
            # Create CompositeTransform for final result with single interpolation
            logger.debug("Creating CompositeTransform for final T1 output")
            combo = sitk.CompositeTransform(3)
            combo.AddTransform(final_rigid_transform)
            combo.AddTransform(final_affine_transform)
            
            # Single resample from original moving image for better quality
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(combo)
            
            final_moved_image = resampler.Execute(moving_image)
            
            # Save the final results (output of two-stage registration)
            # The final warped image is the result of both transforms
            final_warped_path = os.path.join(os.path.dirname(output_prefix), f"{base_name}_mni_warped.nii.gz")
            affine_transform_path = os.path.join(other_dir, f"{base_name}_affine.mat")
            
            sitk.WriteImage(final_moved_image, final_warped_path)
            sitk.WriteTransform(final_affine_transform, affine_transform_path)
            logger.debug(f"Saved final warped image to {final_warped_path}")
            
            # Note: Success tracking moved to main loop after complete processing
            
            # Create a mock result object to maintain compatibility
            class MockResult:
                class Outputs:
                    def __init__(self, prefix, base_name, other_dir):
                        # Final results (two-stage pipeline)
                        self.warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_warped.nii.gz")
                        # Both transforms needed for complete transformation
                        self.forward_transforms = [
                            os.path.join(other_dir, f"{base_name}_rigid.mat"),
                            os.path.join(other_dir, f"{base_name}_affine.mat")
                        ]
                        
                        # Individual stage results for reference
                        self.rigid_warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_rigid_warped.nii.gz")
                        self.rigid_transform = os.path.join(other_dir, f"{base_name}_rigid.mat")
                        self.affine_warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_warped.nii.gz")
                        self.affine_transform = os.path.join(other_dir, f"{base_name}_affine.mat")
                
                def __init__(self, prefix, base_name, other_dir):
                    self.outputs = self.Outputs(prefix, base_name, other_dir)
            
            return MockResult(output_prefix, base_name, other_dir)
            
        except Exception as e:
            logger.error(f"Registration failed with error: {str(e)}")
            raise
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Registration failed for subject {subject_id}: {error_msg}")
        if subject_id:
            registration_results['failed_subjects'].append(subject_id)
            registration_results['errors'][subject_id] = error_msg
        raise

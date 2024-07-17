import argparse
import matplotlib.pyplot as plt
from JoNet_UnitTestModuleFunction import ImageProcessingTester

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run image processing tests.')
    parser.add_argument('--module', '-m', choices=['JoNetUnitTestSplit', 'JoNetUnitTestMin', 'JoNetUnitTestErode', 'JoNetUnitStructElement',
                                                   'JoNetUnitDehaze', 'all'], default='all', help='Module to test or "all" for all modules')
    parser.add_argument('--image-directory', '-i', default='./Test_Image/HaZeFre', help='Path to the directory containing images')
    parser.add_argument('--use-gen', '-g', action='store_true', help='Use randomly generated sample images instead of images from directory')
    parser.add_argument('--num-test', '-n', type=int, default=5, help='Number of tests for the image processing library simulation')

    args = parser.parse_args()
    print(args)
    # Run tests based on command-line arguments
    tester = ImageProcessingTester()
    tester.run_tests(module=args.module,
                     image_directory=args.image_directory, 
                     use_generated_sample=args.use_gen, 
                     num_sample=args.num_test)
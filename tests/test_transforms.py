from image_processing import *


def main():
    width_px = 841
    height_px = 594
    width_mm = 1164
    height_mm = 816

    def print_transform(matrix, pt):
        print(f'{np.array(pt)} -> {transform_affine(matrix, pt)}')

    image_to_world = get_image_to_world_matrix(width_px, height_px, width_mm, height_mm)

    print('Image to world:')
    print_transform(image_to_world, (0, 0))
    print_transform(image_to_world, (width_px, 0))
    print_transform(image_to_world, (0, height_px))
    print_transform(image_to_world, (width_px, height_px))
    print_transform(image_to_world, (width_px // 4, height_px // 4))

    world_to_image = get_world_to_image_matrix(width_mm, height_mm, width_px, height_px)

    print('\nWorld to image:')
    print_transform(world_to_image, (0, 0))
    print_transform(world_to_image, (width_mm, 0))
    print_transform(world_to_image, (0, height_mm))
    print_transform(world_to_image, (width_mm, height_mm))
    print_transform(world_to_image, (width_mm // 4, height_mm // 4))


if __name__ == '__main__':
    main()

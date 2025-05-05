/**
 * @file mnist.h
 * @brief MNIST dataset loading and handling functions
 * 
 * This header provides functionality for loading and managing the MNIST
 * handwritten digits dataset. It handles:
 * - Reading binary IDX format files
 * - Byte order conversion
 * - Image and label data storage
 * - Image saving and visualization
 * 
 * Based on work by Takafumi Hoiruchi (2018)
 * https://github.com/takafumihoriuchi/MNIST_for_C
 */

#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>

/** Magic numbers for file format validation */
#define IMAGE_FILE_MAGIC 0x00000803
#define LABEL_FILE_MAGIC 0x00000801

/** Dataset dimensions */
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define PIXEL_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT)

/** File paths for MNIST dataset */
#define TRAIN_IMAGES_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "data/train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "data/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "data/t10k-labels.idx1-ubyte"

/** Raw image data storage */
unsigned char training_images[60000][PIXEL_COUNT];
unsigned char test_images[10000][PIXEL_COUNT];

/** Ground truth label storage */
unsigned char training_labels[60000];
unsigned char test_labels[10000];

/**
 * @brief Reverses byte order for proper endianness
 * @param value Value to reverse
 * @return Byte-reversed value
 * 
 * MNIST files are stored in big-endian format,
 * this function converts to host byte order
 */
static inline int reverse_byte_order(int value)
{
  unsigned char byte1, byte2, byte3, byte4;
  byte1 = value & 255;
  byte2 = (value >> 8) & 255;
  byte3 = (value >> 16) & 255;
  byte4 = (value >> 24) & 255;
  return ((int)byte1 << 24) + ((int)byte2 << 16) + ((int)byte3 << 8) + byte4;
}

/**
 * @brief Loads image data from IDX file
 * @param filepath Path to IDX file
 * @param image_buffer Array to store image data
 * @param num_images Number of images to read
 * @return 1 on success, 0 on failure
 */
static inline int load_image_file(const char *filepath, unsigned char image_buffer[][PIXEL_COUNT], int num_images)
{
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    printf("Error opening image file: %s\n", filepath);
    return 0;
  }

  int magic_number = 0;
  int image_count = 0;
  int row_count = 0;
  int col_count = 0;

  fread(&magic_number, sizeof(int), 1, file);
  fread(&image_count, sizeof(int), 1, file);
  fread(&row_count, sizeof(int), 1, file);
  fread(&col_count, sizeof(int), 1, file);

  magic_number = reverse_byte_order(magic_number);
  image_count = reverse_byte_order(image_count);
  row_count = reverse_byte_order(row_count);
  col_count = reverse_byte_order(col_count);

  if (magic_number != IMAGE_FILE_MAGIC) {
    printf("Invalid image file format: %s\n", filepath);
    fclose(file);
    return 0;
  }

  fread(image_buffer, sizeof(unsigned char), num_images * PIXEL_COUNT, file);
  fclose(file);
  return 1;
}

/**
 * @brief Loads label data from IDX file
 * @param filepath Path to IDX file
 * @param label_buffer Array to store labels
 * @param num_labels Number of labels to read
 * @return 1 on success, 0 on failure
 */
static inline int load_label_file(const char *filepath, unsigned char *label_buffer, int num_labels)
{
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    printf("Error opening label file: %s\n", filepath);
    return 0;
  }

  int magic_number = 0;
  int label_count = 0;

  fread(&magic_number, sizeof(int), 1, file);
  fread(&label_count, sizeof(int), 1, file);

  magic_number = reverse_byte_order(magic_number);
  label_count = reverse_byte_order(label_count);

  if (magic_number != LABEL_FILE_MAGIC) {
    printf("Invalid label file format: %s\n", filepath);
    fclose(file);
    return 0;
  }

  fread(label_buffer, sizeof(unsigned char), num_labels, file);
  fclose(file);
  return 1;
}

/**
 * @brief Loads complete MNIST dataset
 * @return 1 on success, 0 if any file fails to load
 */
static inline int load_mnist(void)
{
  int success = 1;
  success &= load_image_file(TRAIN_IMAGES_PATH, training_images, 60000);
  success &= load_label_file(TRAIN_LABELS_PATH, training_labels, 60000);
  success &= load_image_file(TEST_IMAGES_PATH, test_images, 10000);
  success &= load_label_file(TEST_LABELS_PATH, test_labels, 10000);
  return success;
}

#endif // MNIST_H

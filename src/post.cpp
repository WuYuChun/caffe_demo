#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>

#define FLOAT_EQUAL_EPSILON 1.0e-6

struct Psd_out {
  std::vector<int> aver_x;
  std::vector<int> aver_y;
  std::vector<uint8_t> aver_sig;
  int seg_id;
  float weight;
};

struct Point {
  int x;
  int y;

  bool operator < (const Point &p) const {
    if (x == p.x) {
      return y < p.y;
    }
    else {
      return x < p.x;
    }
  }
};

uint8_t *load_uint8_data_from_file(const std::string file_name, const int height, const int width) {
  std::ifstream in(file_name.c_str());
  std::string src;

  size_t buf_size = sizeof(uint8_t) * height * width;
  uint8_t *input = (uint8_t *)malloc(buf_size);
  memset(input, 0, buf_size);

  int h = 0, w = 0;
  uint8_t u = 0;
  while (getline(in, src)) {
    std::stringstream ss;
    ss.str(src);
    std::string item;

    w = 0;
    while (getline(ss, item, ',')) {
      u = (uint8_t)atoi(item.c_str());
      input[h * width + w] = u;
      w++;
    }
    h++;
  }

  return input;
}

int *load_int_data_from_file(const std::string file_name, const int channel, const int height, const int width) {
  std::ifstream in(file_name.c_str());
  std::string src;

  size_t buf_size = sizeof(int) * channel * height * width;
  int *input = (int *)malloc(buf_size);
  memset(input, 0, buf_size);

  int h = 0, w = 0;
  int i = 0;
  while (getline(in, src)) {
    std::stringstream ss;
    ss.str(src);
    std::string item;

    w = 0;
    while (getline(ss, item, ',')) {
      i = (int)atoi(item.c_str());
      input[h * width + w] = i;
      w++;
    }
    h++;
  }

  return input;
}

float *load_float_data_from_file(const std::string file_name, const int channel, const int height, const int width) {
  std::ifstream in(file_name.c_str());
  std::string src;

  size_t buf_size = sizeof(float) * channel * height * width;
  float *input = (float *)malloc(buf_size);
  memset(input, 0.0f, buf_size);

  int h = 0, w = 0;
  float f = 0.0f;
  while (getline(in, src)) {
    std::stringstream ss;
    ss.str(src);
    std::string item;

    w = 0;
    while (getline(ss, item, ',')) {
      f = atof(item.c_str());
      input[h * width + w] = f;
      w++;
    }
    h++;
  }

  return input;
}

int binarize_seg(const uint8_t *input_seg, const int height, const int width, uint8_t *findcc_input) {
  for (int i = 0; i < height * width; ++i) {
    if (input_seg[i] == 1 || input_seg[i] == 2) {
      findcc_input[i] = 1;
    }
    else if (input_seg[i] == 0) {
      findcc_input[i] = 0;
    }
    else { // input_seg data validation check
      printf("Error: input_seg[%d] is %u, should be 0, 1, or 2!\n", i, input_seg[i]);
      return -1;
    }
  }
  return 0;
}

int dx[4] = {-1, 0, 1, 0};
int dy[4] = {0, -1, 0, 1};

void dfs(const uint8_t *findcc_input, int *findcc_output, const int height, const int width, int x, int y, int c) {
  findcc_output[y * width + x] = c;
  for (int i = 0; i < 4; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    if (nx < 0 || nx > width - 1 || ny < 0 || ny > height - 1) {
      continue;
    }
    if (findcc_input[ny * width + nx] > 0 && findcc_output[ny * width + nx] < 1) {
      dfs(findcc_input, findcc_output, height, width, nx, ny, c);
    }
  }
}

int find_connected_components(const uint8_t *findcc_input, const int height, const int width, int *findcc_output) {
  int cur_label = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      if (findcc_input[h * width + w] > 0 && findcc_output[h * width + w] < 1) {
        dfs(findcc_input, findcc_output, height, width, w, h, ++cur_label);
      }
    }
  }
  return cur_label;
}

// axis = 0 indicates x direction --> width
// axis = 1 indicates y direction --> height
int update_coordinate(float *input, const int channel_num, const int height, const int width, const int threshold, const int axis) {
  int index = 0;
  for (int i = 0; i < channel_num * height * width; ++i) {
    index = i % (height * width);
    if (axis == 0) { // x direction --> width
      index = index % width;
    }
    else { // y direction --> height
      index = index / width;
    }
    input[i] = (int)(input[i] + index);

    if (input[i] < 0) {
      input[i] = 0;
    }
    else if (input[i] > threshold - 1) {
      input[i] = threshold - 1;
    }
  }

  return 0;
}

bool compare(const std::pair<Point, int> &p_a, const std::pair<Point, int> &p_b) {
  if (p_a.second == p_b.second) {
    if (p_a.first.x == p_b.first.x) {
      return p_a.first.y > p_b.first.y;
    }
    else {
      return p_a.first.x > p_b.first.x;
    }
  }
  else {
    return p_a.second > p_b.second;
  }
}

int calc_psd_pos(const int num_cc, const int channel, const int height, const int width, const float sig_threshold, const unsigned int ignored_cc_pixel_num, const uint8_t *input_seg, const int *findcc_output, const float *coord_x, const float *coord_y, const float *input_sigmoid, std::vector<Psd_out> &psd, std::vector<float> &time_vec) {
  /*for (int i = 0; i < channel * height * width; ++i) {
    if (coord_x[i] < 0 || coord_x[i] > width - 1) {
      printf("[calc_psd_pos] error coord_x[%d] is %f!\n", i, coord_x[i]);
    }
    if (coord_y[i] < 0 || coord_y[i] > height - 1) {
      printf("[calc_psd_pos] error coord_y[%d] is %f!\n", i, coord_y[i]);
    }
  }*/

  float time_f[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::chrono::high_resolution_clock::time_point start, end;

  for (int cc_id = 1; cc_id <= num_cc; ++cc_id) {
    printf("cc %d\n", cc_id);

    // iterate over each cc
    std::vector<int> aver_x_vec;
    std::vector<int> aver_y_vec;
    std::vector<uint8_t> aver_sig_vec;
    int seg_id = 0;
    float weight = 0.0f;

    start = std::chrono::high_resolution_clock::now();

    std::vector<int> cc_index_vec;
    for (int cc_index = 0; cc_index < height * width; ++cc_index) {
      if (findcc_output[cc_index] == cc_id) {
        cc_index_vec.push_back(cc_index);
      }
    }
    assert(cc_index_vec.size() > 0);
    printf("cc_index_vec.size() is %zu\n", cc_index_vec.size());

    end = std::chrono::high_resolution_clock::now();
    auto duration_f = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    time_f[0] += static_cast<float>(duration_f) / 1000.0f;

    if (cc_index_vec.size() < ignored_cc_pixel_num) continue;

    float time_c[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int channel_id = 0; channel_id < channel; ++channel_id) {
      printf("channel %d\n", channel_id);
      int offset = channel_id * height * width;
      std::vector<int> coord_x_vec;
      std::vector<int> coord_y_vec;

      start = std::chrono::high_resolution_clock::now();

      for (unsigned int i = 0; i < cc_index_vec.size(); ++i) {
        coord_x_vec.push_back((int)coord_x[offset + cc_index_vec[i]]);
        coord_y_vec.push_back((int)coord_y[offset + cc_index_vec[i]]);
      }

      assert(coord_x_vec.size() > 0);

      end = std::chrono::high_resolution_clock::now();
      auto duration_c = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      time_c[0] += static_cast<float>(duration_c) / 1000.0f;
      start = std::chrono::high_resolution_clock::now();

      std::map<Point, int> point_map;
      for (unsigned int i = 0; i < coord_x_vec.size(); ++i) {
	Point p = { coord_x_vec[i], coord_y_vec[i] };
	std::map<Point, int>::iterator iter = point_map.find(p);
	if (iter != point_map.end()) {
	  iter->second++;
	}
	else {
	  point_map.insert(std::pair<Point, int>(p, 1));
	}
      }

      end = std::chrono::high_resolution_clock::now();
      duration_c = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      time_c[1] += static_cast<float>(duration_c) / 1000.0f;
      start = std::chrono::high_resolution_clock::now();

      std::vector<std::pair<Point, int> > v(point_map.begin(), point_map.end());
      sort(v.begin(), v.end(), compare);

      end = std::chrono::high_resolution_clock::now();
      duration_c = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      time_c[2] += static_cast<float>(duration_c) / 1000.0f;
      start = std::chrono::high_resolution_clock::now();

      int count_max = v[0].second;
      printf("count_max is %d\n", count_max);
      std::vector<Point> max_points;
      for (unsigned int i = 0; i < v.size(); ++i) {
	if (v[i].second == count_max) {
	  max_points.push_back(v[i].first);
	}
	else {
	  break;
	}
      }

      assert(max_points.size() > 0);
      printf("max_points.size() is %zu\n", max_points.size());
      
      end = std::chrono::high_resolution_clock::now();
      duration_c = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      time_c[3] += static_cast<float>(duration_c) / 1000.0f;
      start = std::chrono::high_resolution_clock::now();

      // calculate average coordinate
      float x_coord_aver = 0.0f;
      float y_coord_aver = 0.0f;
      for (unsigned int i = 0; i < max_points.size(); ++i) {
	x_coord_aver += max_points[i].x;
	y_coord_aver += max_points[i].y;
      }
      x_coord_aver /= max_points.size();
      y_coord_aver /= max_points.size();

      // calculate average sigmoid value
      float aver_sig = 0.0f;
      int sig_count = 0;
      for (unsigned int i = 0; i < max_points.size(); ++i) {
	for (unsigned int j = 0; j < cc_index_vec.size(); ++j) {
          if ((int)coord_x[offset + cc_index_vec[j]] == max_points[i].x && (int)coord_y[offset + cc_index_vec[j]] == max_points[i].y) {
	    aver_sig += input_sigmoid[offset + cc_index_vec[j]];
	    sig_count++;
	  }
	}
      }
      assert(sig_count > 0);
      printf("aver_sig is %f, sig_count is %d\n", aver_sig, sig_count);
      aver_sig /= sig_count;
      uint8_t aver_sig_uint8 = (aver_sig > sig_threshold) ? 0 : 1;

      aver_x_vec.push_back((int)x_coord_aver);
      aver_y_vec.push_back((int)y_coord_aver);
      aver_sig_vec.push_back(aver_sig_uint8);

      end = std::chrono::high_resolution_clock::now();
      duration_c = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      time_c[4] += static_cast<float>(duration_c) / 1000.0f;
    }

    time_f[1] += time_c[0];
    time_f[2] += time_c[1];
    time_f[3] += time_c[2];
    time_f[4] += time_c[3];
    time_f[5] += time_c[4];

    start = std::chrono::high_resolution_clock::now();

    int count_one = 0, count_two = 0;
    for (unsigned int i = 0; i < cc_index_vec.size(); ++i) {
      if (input_seg[cc_index_vec[i]] == 1) {
	count_one++;
      }
      else if (input_seg[cc_index_vec[i]] == 2) {
	count_two++;
      }
      else {
	printf("Error: value of input_seg[%d] is %d, while findcc_output[%d] is %u!\n", cc_index_vec[i], input_seg[cc_index_vec[i]], cc_index_vec[i], findcc_output[cc_index_vec[i]]);
	return -1;
      }
    }
    assert(count_one + count_two > 0);
    int max_count = 0;
    if (count_one > count_two) {
      seg_id = 1;
      max_count = count_one;
    }
    else {
      seg_id = 2;
      max_count = count_two;
    }
    weight = (float)max_count / (count_one + count_two);

    end = std::chrono::high_resolution_clock::now();
    duration_f = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    time_f[6] += static_cast<float>(duration_f) / 1000.0f;

    Psd_out temp_psd_out = { aver_x_vec, aver_y_vec, aver_sig_vec, seg_id, weight };
    psd.push_back(temp_psd_out);
  }

  for (int i = 0; i < 7; i++) {
    time_vec.push_back(time_f[i]);
  }

  return 0;
}

int parse_psd_out_file(const std::string file_name, std::vector<Psd_out> &psd) {
  std::ifstream in(file_name.c_str());
  std::string src;

  int line_count = 0;
  std::vector<std::string> str_vec;
  while (getline(in, src)) {
    str_vec.push_back(src);
    line_count++;

    // process every 5 lines
    if (line_count % 5 == 0) {
      std::vector <int> aver_x;
      std::vector <int> aver_y;
      std::vector <uint8_t> aver_sig;
      int seg_id = 0;
      float weight = 0.0f;

      std::stringstream ss;
      std::string item;

      ss.str(str_vec[0]);
      while (getline(ss, item, ',')) {
	aver_x.push_back(atoi(item.c_str()));
      }

      ss.clear();
      ss.str("");
      ss.str(str_vec[1]);
      while (getline(ss, item, ',')) {
        aver_y.push_back(atoi(item.c_str()));
      }

      ss.clear();
      ss.str("");
      ss.str(str_vec[2]);
      while (getline(ss, item, ',')) {
        aver_sig.push_back((uint8_t)atoi(item.c_str()));
      }

      seg_id = atoi(str_vec[3].c_str());
      weight = atof(str_vec[4].c_str());

      Psd_out temp_psd_out = { aver_x, aver_y, aver_sig, seg_id, weight };
      psd.push_back(temp_psd_out);

      std::vector<std::string>().swap(str_vec); // reset the vector
      line_count = 0;
    }
  }

  return 0;
}

int verify_psd_out(const std::vector<Psd_out> psd, const std::vector<Psd_out> psd_truth) {
  if (psd.size() != psd_truth.size()) {
    printf("[VERIFY] Failed! psd.size is %zu, but psd_truth.size is %zu!\n", psd.size(), psd_truth.size());
    return -1;
  }

  for (unsigned int i = 0; i < psd.size(); ++i) {
    // verify aver_x
    if (psd[i].aver_x.size() != psd_truth[i].aver_x.size()) {
      printf("[VERIFY] Failed! psd[%d].aver_x.size is %zu, but psd_truth[%d].aver_x.size is %zu!\n", i, psd[i].aver_x.size(), i, psd_truth[i].aver_x.size());
      return -1;
    }

    for (unsigned int j = 0; j < psd[i].aver_x.size(); ++j) {
      if (psd[i].aver_x[j] != psd_truth[i].aver_x[j]) {
	printf("[VERIFY] Failed! psd[%d].aver_x[%d] is %d, but psd_truth[%d].aver_x[%d] is %d!\n", i, j, psd[i].aver_x[j], i, j, psd_truth[i].aver_x[j]);
	return -1;
      }
    }

    // verify aver_y
    if (psd[i].aver_y.size() != psd_truth[i].aver_y.size()) {
      printf("[VERIFY] Failed! psd[%d].aver_y.size is %zu, but psd_truth[%d].aver_y.size is %zu!\n", i, psd[i].aver_y.size(), i, psd_truth[i].aver_y.size());
      return -1;
    }

    for (unsigned int j = 0; j < psd[i].aver_y.size(); ++j) {
      if (psd[i].aver_y[j] != psd_truth[i].aver_y[j]) {
        printf("[VERIFY] Failed! psd[%d].aver_y[%d] is %d, but psd_truth[%d].aver_y[%d] is %d!\n", i, j, psd[i].aver_y[j], i, j, psd_truth[i].aver_y[j]);
        return -1;
      }
    }

    // verify aver_sig
    if (psd[i].aver_sig.size() != psd_truth[i].aver_sig.size()) {
      printf("[VERIFY] Failed! psd[%d].aver_sig.size is %zu, but psd_truth[%d].aver_sig.size is %zu!\n", i, psd[i].aver_sig.size(), i, psd_truth[i].aver_sig.size());
      return -1;
    }

    for (unsigned int j = 0; j < psd[i].aver_sig.size(); ++j) {
      if (psd[i].aver_sig[j] != psd_truth[i].aver_sig[j]) {
        printf("[VERIFY] Failed! psd[%d].aver_sig[%d] is %d, but psd_truth[%d].aver_sig[%d] is %d!\n", i, j, psd[i].aver_sig[j], i, j, psd_truth[i].aver_sig[j]);
        return -1;
      }
    }

    // verify seg_id
    if (psd[i].seg_id != psd_truth[i].seg_id) {
      printf("[VERIFY] Failed! psd[%d].seg_id is %d, but psd_truth[%d].seg_id is %d!\n", i, psd[i].seg_id, i, psd_truth[i].seg_id);
      return -1;
    }

    // verify weight
    if (fabs(psd[i].weight - psd_truth[i].weight) > FLOAT_EQUAL_EPSILON) {
      printf("[VERIFY] Failed! psd[%d].weight is %f, but psd_truth[%d].weight is %f!\n", i, psd[i].weight, i, psd_truth[i].weight);
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  int err = -1;

  std::chrono::high_resolution_clock::time_point program_start, program_end;
  std::chrono::high_resolution_clock::time_point post_start, post_end;
  std::chrono::high_resolution_clock::time_point start, end;

  double load_data_time_ms = 0.0f;
  double input_data_preparation_time_ms = 0.0f;
  double binarize_seg_time_ms = 0.0f;
  double findcc_time_ms = 0.0f;
  double update_coord_x_time_ms = 0.0f;
  double update_coord_y_time_ms = 0.0f;
  double psd_time_ms = 0.0f;
  double post_time_ms = 0.0f;
  double program_time_ms = 0.0f;

  program_start = std::chrono::high_resolution_clock::now();
  start = program_start;

  // process input parameters and load data
  int height = 1040;//2080;
  int width = 640;//1280;
  int channel_num = 4;

  std::string seg_file_name = argv[1];
  std::string coord_x_file_name = argv[2];
  std::string coord_y_file_name = argv[3];
  std::string sigmoid_file_name = argv[4];

  int is_verify = atoi(argv[5]);
  int is_profiling = atoi(argv[6]);
  //int is_save_result = atoi(argv[7]);
  std::string file_number = seg_file_name.substr(seg_file_name.find_last_of("/") + 1, seg_file_name.length());
  file_number = file_number.substr(0, file_number.find("_"));
  printf("File No. %d\n", atoi(file_number.c_str()));

  uint8_t *input_seg = load_uint8_data_from_file(seg_file_name, height, width);
  float *input_coord_x = load_float_data_from_file(coord_x_file_name, channel_num, height, width);
  float *input_coord_y = load_float_data_from_file(coord_y_file_name, channel_num, height, width);
  float *input_sigmoid = load_float_data_from_file(sigmoid_file_name, channel_num, height, width);
  printf("Input data loaded.\n");

  end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  load_data_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();

  // input data preparation
  size_t uint8_buf_size = sizeof(uint8_t) * height * width;
  size_t int_buf_size = sizeof(int) * height * width;
  //size_t float_buf_size = sizeof(float) * height * width;

  uint8_t *findcc_input = (uint8_t *)malloc(uint8_buf_size);
  memset(findcc_input, 0, uint8_buf_size);

  int *findcc_output = (int *)malloc(int_buf_size);
  memset(findcc_output, 0, int_buf_size);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  input_data_preparation_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();
  post_start = start;

  // binarize input_seg
  err = binarize_seg(input_seg, height, width, findcc_input);
  assert(err == 0);
  printf("Binarize input_seg success.\n");

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  binarize_seg_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();

  // findcc
  int num_cc = find_connected_components(findcc_input, height, width, findcc_output);
  printf("Findcc over. %d connected component(s) found.\n", num_cc);
  assert(num_cc > 0);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  findcc_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();

  // update cooord_x
  err = update_coordinate(input_coord_x, channel_num, height, width, width, 0);
  assert(err == 0);
  printf("Coord_x updated.\n");

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  update_coord_x_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();

  // update cooord_y
  err = update_coordinate(input_coord_y, channel_num, height, width, height, 1);
  assert(err == 0);
  printf("Coord_y updated.\n");

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  update_coord_y_time_ms = static_cast<float>(duration) / 1000.0f;
  start = std::chrono::high_resolution_clock::now();

  // psd calculation
  float sig_threshold = 0.2f;
  unsigned int ignored_cc_pixel_num = 100;
  std::vector<Psd_out> psd;
  std::vector<float> psd_time_vec;
  err = calc_psd_pos(num_cc, channel_num, height, width, sig_threshold, ignored_cc_pixel_num, input_seg, findcc_output, input_coord_x, input_coord_y, input_sigmoid, psd, psd_time_vec);
  assert(err == 0);
  printf("PSD calculation over.\n");

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  psd_time_ms = static_cast<float>(duration) / 1000.0f;

  post_end = end;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count();
  post_time_ms = static_cast<float>(duration) / 1000.0f;

  program_end = end;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();
  program_time_ms = static_cast<float>(duration) / 1000.0f;

  printf("PSD results:\n");
  for (unsigned int i = 0; i < psd.size(); ++i) {
    printf("====================\n%d\n", i);
    for (unsigned int j = 0; j < psd[i].aver_x.size(); ++j) {
      printf("%d ", psd[i].aver_x[j]);
    }
    printf("\n");
    for (unsigned int j = 0; j < psd[i].aver_y.size(); ++j) {
      printf("%d ", psd[i].aver_y[j]);
    }
    printf("\n");
    for (unsigned int j = 0; j < psd[i].aver_sig.size(); ++j) {
      printf("%u ", psd[i].aver_sig[j]);
    }
    printf("\n");
    printf("%d\n%f\n===================\n", psd[i].seg_id, psd[i].weight);
  }

  if (is_verify) {
    std::vector<Psd_out> psd_truth;
    std::string psd_out_file_name = argv[8];
    err = parse_psd_out_file(psd_out_file_name, psd_truth);
    assert(err == 0);

    /*printf("PSD_TRUTH results:\n");
    for (unsigned int i = 0; i < psd_truth.size(); ++i) {
      printf("====================\n%d\n", i);
      for (unsigned int j = 0; j < psd_truth[i].aver_x.size(); ++j) {
        printf("%d ", psd_truth[i].aver_x[j]);
      }
      printf("\n");
      for (unsigned int j = 0; j < psd_truth[i].aver_y.size(); ++j) {
        printf("%d ", psd_truth[i].aver_y[j]);
      }
      printf("\n");
      for (unsigned int j = 0; j < psd_truth[i].aver_sig.size(); ++j) {
        printf("%u ", psd_truth[i].aver_sig[j]);
      }
      printf("\n");
      printf("%d\n%f\n===================\n", psd_truth[i].seg_id, psd_truth[i].weight);
    }*/

    err = verify_psd_out(psd, psd_truth);
    if (err != 0) {
      std::ofstream verify_error_stream("verify_error_info.txt", std::ofstream::out | std::ofstream::app);
      verify_error_stream << file_number << "\tpsd_out verified wrong\n";
      verify_error_stream.close();
    }
    assert(err == 0);
    printf("[VERIFY] PSD out is correct.\n");
  }

  free(input_seg);
  free(input_coord_x);
  free(input_coord_y);
  free(input_sigmoid);
  free(findcc_input);
  free(findcc_output);

  printf("\n=============== PROFILING INFO (ms) ===============\n");
  printf("[load_input_data]               %f\n", load_data_time_ms);
  printf("[input_data_preparation]        %f\n", input_data_preparation_time_ms);
  printf("[binarize_seg]                  %f\n", binarize_seg_time_ms);
  printf("[findcc]                        %f\n", findcc_time_ms);
  printf("[update_coord_x]                %f\n", update_coord_x_time_ms);
  printf("[update_coord_y]                %f\n", update_coord_y_time_ms);	
  printf("[calc_psd]                      %f\n", psd_time_ms);
  for (unsigned int i = 0; i < psd_time_vec.size(); ++i) {
    printf("[calc_psd %d]                    %f\n", i, psd_time_vec[i]);
  }
  printf("\n");
  printf("[post_time]                     %f\n", post_time_ms);
  printf("[TOTAL]                         %f\n", program_time_ms);
  printf("===================================================\n");
    
  if (is_profiling) {
    std::ofstream out_stream("profiling.txt", std::ofstream::out | std::ofstream::app);
		
    out_stream << file_number << "\t";
    out_stream << load_data_time_ms << "\t" << input_data_preparation_time_ms << "\t";
    out_stream << binarize_seg_time_ms << "\t";
    out_stream << findcc_time_ms << "\t";
    out_stream << update_coord_x_time_ms << "\t" << update_coord_y_time_ms << "\t";
    out_stream << psd_time_ms << "\t";
    out_stream << psd_time_vec[0] << "\t";
    out_stream << psd_time_vec[1] << "\t";
    out_stream << psd_time_vec[2] << "\t";
    out_stream << psd_time_vec[3] << "\t";
    out_stream << psd_time_vec[4] << "\t";
    out_stream << psd_time_vec[5] << "\t";
    out_stream << psd_time_vec[6] << "\t";
    out_stream << post_time_ms << "\t" << program_time_ms << std::endl;
    out_stream.close();
  }

  return 0;
}

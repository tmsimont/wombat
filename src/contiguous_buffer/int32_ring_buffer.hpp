#ifndef CONTIGUOUS_BUFFER_INT32_RING_BUFFER_H_
#define CONTIGUOUS_BUFFER_INT32_RING_BUFFER_H_

#include <stdlib.h>
#include <memory>
#include <array>

namespace wombat {

  /**
   * A ring buffer that is backed by a contiguous block of memory.
   * The underlying data structure is a big array of integers.
   * Entries in the buffer are stored as sub-arrays of the contiguous integer block.
   * Entries should have some semantic meaning behind the sub-array.
   * Each entry is the same length sub-array.
   * The template param S is the per-item size in number of integers.
   */
  template <int S>
  class Int32RingBuffer {
    public:
      Int32RingBuffer(int32_t num_items) {
        _num_items = num_items;
        posix_memalign(
            reinterpret_cast<void **>(&_data),
            64,
            S * _num_items * sizeof(int32_t));
      }

      ~Int32RingBuffer() {
        free(_data);
      }

      bool isFull() {
        return i_empty == i_ready && i_empty_wrap != i_ready_wrap;
      }

      bool isEmpty() {
        return i_empty == i_ready && i_empty_wrap == i_ready_wrap;
      }

      int32_t itemSize() {
        return S;
      }

      int32_t numItems() {
        return _num_items;
      }

      /**
       * User passes in a array of ints that backs the entry, and it will be copied
       * into the buffer. The input array is discarded/destroyed.
       */
      int32_t push(std::unique_ptr<std::array<int, S>> data) {
        if (isFull()) return 0;
        std::copy(data->begin(), data->end(), _data + S * i_empty);
        i_empty++;
        if (i_empty == _num_items) {
          i_empty = 0;
          ++i_empty_wrap;
          i_ready_wrap = i_ready_wrap - i_empty_wrap;
          i_empty_wrap = 0;
        }
        return 1;
      }

      /**
       * User gets a unique pointer to a new array that has the buffer contents copied
       * out. The entry in the buffer is free for future use since the data has been
       * copied.
       */
      std::unique_ptr<std::array<int, S>> pop() {
        if (isEmpty()) return nullptr;
        std::unique_ptr<std::array<int, S>> item = 
          std::make_unique<std::array<int, S>>(std::array<int, S>());
        int32_t *ptr = _data + i_ready * S;
        for (int i = 0; i < S; ++i) {
          (*item)[i] = *ptr++;
        }
        i_ready++;
        if (i_ready == _num_items) {
          i_ready = 0;
          ++i_ready_wrap;
          i_ready_wrap = i_ready_wrap - i_empty_wrap;
          i_empty_wrap = 0;
        }
        return item;
      }

    private:
      int32_t i_empty = 0, i_empty_wrap = 0;
      int32_t i_ready = 0, i_ready_wrap = 0;
      int32_t _num_items;
      int32_t *_data;
  };
}

#endif

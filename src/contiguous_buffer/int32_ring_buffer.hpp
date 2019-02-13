#ifndef CONTIGUOUS_BUFFER_INT32_RING_BUFFER_H_
#define CONTIGUOUS_BUFFER_INT32_RING_BUFFER_H_

#include <stdlib.h>

#include <memory>
#include <sstream>
#include <vector>

namespace wombat {

  /**
   * A ring buffer that is backed by a contiguous block of memory.
   * The underlying data structure is a big array of integers.
   * Entries in the buffer are stored as sub-arrays of the contiguous integer block.
   * Entries should have some semantic meaning behind the sub-array.
   * Each entry is the same length sub-array.
   */
  class Int32RingBuffer {
    public:
      Int32RingBuffer(int32_t num_items, int32_t entry_size) {
        _num_items = num_items;
        _entry_size = entry_size;
        posix_memalign(
            reinterpret_cast<void **>(&_data),
            64,
            _entry_size * _num_items * sizeof(int32_t));
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
        return _entry_size;
      }

      int32_t numItems() {
        return _num_items;
      }

      /**
       * User passes in shared pointer to a vector of ints that backs the entry.
       * It will be de-referenced and copied into the buffer, or ignored if the buffer
       * is full.
       * @return 0 on full buffer or 1 on successful insertion.
       * @return -1 on data vector of incorrect size.
       */
      int32_t push(std::shared_ptr<std::vector<int>> data) {
        if (isFull()) return 0;
        if (data->size() != _entry_size) {
          std::stringstream ss;
          ss << "Data vector passed to ring buffer is incorrect size.";
          ss << " given " << data->size() << ", expected " << _entry_size;
          throw std::invalid_argument(ss.str());
        }
        std::copy(data->begin(), data->end(), _data + _entry_size * i_empty);
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
       * User gets a unique pointer to a new vector that has the buffer contents copied
       * out. The entry in the buffer is free for future use since the data has been
       * copied.
       */
      std::unique_ptr<std::vector<int>> pop() {
        if (isEmpty()) return nullptr;
        std::unique_ptr<std::vector<int>> item = std::make_unique<std::vector<int>>(_entry_size);
        std::copy(_data + i_ready * _entry_size, _data + (i_ready + 1) * _entry_size, item->begin());
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
      int32_t _entry_size;
      int32_t *_data;
  };
}

#endif

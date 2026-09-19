#include "actlab/sha256.hpp"

#include "actlab_test.hpp"

#include <filesystem>
#include <fstream>

ACTLAB_TEST(sha256_known_vectors) {
  CHECK_EQ(actlab::sha256_hex(""),
           std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(actlab::sha256_hex("abc"),
           std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  CHECK_EQ(actlab::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
}

ACTLAB_TEST(sha256_file_matches_string) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "actlab_sha256_test.txt";
  {
    std::ofstream out(path, std::ios::trunc);
    out << "abc";
  }
  CHECK_EQ(actlab::sha256_file(path), actlab::sha256_hex("abc"));
  std::filesystem::remove(path);
}

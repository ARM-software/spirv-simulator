#include "util.hpp"

#include <fstream>

namespace util
{
	std::vector<uint32_t> ReadFile(const std::string &path)
	{
	    std::ifstream infile(path, std::ios::binary | std::ios::ate);

	    if(!infile){
	        throw std::runtime_error("Cannot open " + path);
	    }

	    auto size = infile.tellg();
	    infile.seekg(0, std::ios::beg);

	    std::vector<uint32_t> buf(static_cast<size_t>(size) / 4);

	    infile.read(reinterpret_cast<char*>(buf.data()), size);
	    if (infile.fail()){
	        throw std::runtime_error("Read error");
	    }

	    return buf;
	}
}
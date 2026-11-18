# Embeds INPUT as the byte array NAME (with NAME_size) in the header OUTPUT.
file(READ "${INPUT}" content HEX)
string(REGEX REPLACE "(................................)" "\\1\n" content "${content}")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," content "${content}")
file(WRITE "${OUTPUT}" "#pragma once\n// Generated from ${INPUT}. Do not edit.\n#include <cstddef>\nnamespace tidal::server {\ninline constexpr unsigned char ${NAME}[] = {\n${content}\n};\ninline constexpr std::size_t ${NAME}_size = sizeof ${NAME};\n}\n")

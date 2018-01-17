FILE(REMOVE_RECURSE
  "CMakeFiles/cleanAll"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/cleanAll.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)

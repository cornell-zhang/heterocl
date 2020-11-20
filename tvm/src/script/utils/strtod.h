#pragma once

namespace torch {
namespace jit {

double strtod_c(const char* nptr, char** endptr);
float strtof_c(const char* nptr, char** endptr);

}} // namespace jit, torch

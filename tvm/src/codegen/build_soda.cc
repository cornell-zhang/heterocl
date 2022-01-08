/*!
 *  Copyright (c) 2019 by Contributors
 */
#include "build_soda.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <unistd.h>
#include <fstream>
#include "./build_common.h"
#include "./codegen_soda.h"

namespace TVM {
namespace codegen {

enum class SodaBackend {
  DSL,
  XHLS,
};

void SODA2HLSC(std::string& code) {
  // Handle concatenated code recursively.
  size_t sep = code.find("\n\n");
  if (sep != std::string::npos) {
    std::string code1 = code.substr(0, sep + 1);
    std::string code2 = code.substr(sep + 2);
    SODA2HLSC(code1);
    SODA2HLSC(code2);
    code = code1 + code2;
    return;
  }

  // Check that python3 and sodac are there
  if (system("which python3 >/dev/null") != 0) {
    LOG(WARNING) << "python3 not found";
  }
  if (system("python3 -m soda.sodac -h >/dev/null") != 0) {
    LOG(WARNING) << "sodac not found";
  }

  // Invoke sodac

  // Create pipes for inter-process communication
  int pipe0[2];
  int pipe1[2];
  int pipe2[2];
  CHECK_EQ(pipe(pipe0), 0) << strerror(errno);
  CHECK_EQ(pipe(pipe1), 0) << strerror(errno);
  CHECK_EQ(pipe(pipe2), 0) << strerror(errno);

  // Fork to prepare for inter-process communication
  pid_t pid = fork();
  CHECK_NE(pid, -1) << strerror(errno);
  if (pid) {  // Parent process
    // Close unused read end of pipe0 and write ends of pipe1 & pipe2
    CHECK_EQ(close(pipe0[0]), 0) << strerror(errno);
    CHECK_EQ(close(pipe1[1]), 0) << strerror(errno);
    CHECK_EQ(close(pipe2[1]), 0) << strerror(errno);

    // Write SODA DSL to the write end of pipe0
    CHECK_EQ(write(pipe0[1], code.c_str(), code.size()), code.size())
        << strerror(errno);

    // Close write end of pipe0 to generate EOF
    CHECK_EQ(close(pipe0[1]), 0) << strerror(errno);

    // Open the read ends of pipe1 & pipe2
    std::ifstream stream1("/proc/self/fd/" + std::to_string(pipe1[0]));
    std::ifstream stream2("/proc/self/fd/" + std::to_string(pipe2[0]));

    // Close the old fds of the read ends of pipe1 & pipe2
    CHECK_EQ(close(pipe1[0]), 0) << strerror(errno);
    CHECK_EQ(close(pipe2[0]), 0) << strerror(errno);

    // Read pipe1 & pipe2
    using InputIter = std::istreambuf_iterator<char>;
    std::string content1((InputIter(stream1)), InputIter());
    std::string content2((InputIter(stream2)), InputIter());

    // Use child's stdout as the code output
    code = content1;

    // Use child's stderr as logging messages
    if (!content2.empty()) {
      LOG(INFO) << content2;
    }

    int wstatus;
    wait(&wstatus);
    CHECK_EQ(WEXITSTATUS(wstatus), 0) << "SODA failed";
  } else {  // Child process
    // Close unused write end of pipe0 and read ends of pipe1 & pipe2
    CHECK_EQ(close(pipe0[1]), 0) << strerror(errno);
    CHECK_EQ(close(pipe1[0]), 0) << strerror(errno);
    CHECK_EQ(close(pipe2[0]), 0) << strerror(errno);

    // Replace stdin, stdout, and stderr with pipe0, pipe1, and pipe2
    CHECK_EQ(dup2(pipe0[0], 0), 0) << strerror(errno);
    CHECK_EQ(dup2(pipe1[1], 1), 1) << strerror(errno);
    CHECK_EQ(dup2(pipe2[1], 2), 2) << strerror(errno);

    // Close old fds of pipe0, pipe1, and pipe2
    CHECK_EQ(close(pipe0[0]), 0) << strerror(errno);
    CHECK_EQ(close(pipe1[1]), 0) << strerror(errno);
    CHECK_EQ(close(pipe2[1]), 0) << strerror(errno);

    // Invoke sodac
    CHECK_EQ(execlp("python3", "python3", "-m", "soda.sodac", "--xocl-kernel",
                    "-", "-", nullptr),
             0)
        << strerror(errno);
  }
}

std::string BuildSODA(Array<LoweredFunc> funcs, SodaBackend backend) {
  using TVM::runtime::Registry;
  bool output_ssa = false;
  CodeGenSODA cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_soda_postproc")) {
    code = (*f)(code).operator std::string();
  }

  if (backend == SodaBackend::XHLS) {
    SODA2HLSC(code);
  }

  LOG(WARNING) << "SODA doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_soda")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = BuildSODA(args[0], SodaBackend::DSL);
    });

TVM_REGISTER_API("codegen.build_soda_xhls")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = BuildSODA(args[0], SodaBackend::XHLS);
    });
}  // namespace codegen
}  // namespace TVM

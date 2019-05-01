#include <fstream>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_soda.h"
#include "./build_common.h"
#include "./build_soda.h"

namespace TVM {
namespace codegen {

enum class SodaBackend {
  DSL,
  XHLS
};

void SODA2HLSC(std::string& code) {
  // Mangle PATH to find sodac
  if (char* pythonpath = getenv("PYTHONPATH")) {
    char* path = strtok(pythonpath, ":");
    while (path != nullptr) {
      setenv("PATH",
          (std::string(path) + "/../soda/src:" + getenv("PATH")).c_str(),
          /* overwrite = */1);
      path = strtok(nullptr, ":");
    }
  }

  // Check that python3 and sodac are there
  if (system("which python3 >/dev/null") != 0) {
    LOG(WARNING) << "python3 not found";
  }
  if (system("which sodac >/dev/null") != 0) {
    LOG(WARNING) << "sodac not found";
  }

  // Invoke sodac
  auto check = [](int returned, int expected = 0) {
    if (returned != expected) {
      LOG(WARNING) << strerror(errno);
      exit(errno);
    }
  };

  // Create pipes for inter-process communication
  int pipe0[2];
  int pipe1[2];
  int pipe2[2];
  check(pipe(pipe0));
  check(pipe(pipe1));
  check(pipe(pipe2));

  // Fork to prepare for inter-process communication
  pid_t pid = fork();
  if (pid == -1) { LOG(WARNING) << strerror(errno); }
  if (pid) {  // Parent process
    // Close unused read end of pipe0 and write ends of pipe1 & pipe2
    check(close(pipe0[0]));
    check(close(pipe1[1]));
    check(close(pipe2[1]));

    // Write SODA DSL to the write end of pipe0
    check(write(pipe0[1], code.c_str(), code.size()), code.size());

    // Close write end of pipe0 to generate EOF
    check(close(pipe0[1]));

    // Open the read ends of pipe1 & pipe2
    std::ifstream stream1("/proc/self/fd/" + std::to_string(pipe1[0]));
    std::ifstream stream2("/proc/self/fd/" + std::to_string(pipe2[0]));

    // Close the old fds of the read ends of pipe1 & pipe2
    check(close(pipe1[0]));
    check(close(pipe2[0]));

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

    wait(nullptr);
  } else {  // Child process
    // Close unused write end of pipe0 and read ends of pipe1 & pipe2
    check(close(pipe0[1]));
    check(close(pipe1[0]));
    check(close(pipe2[0]));

    // Replace stdin, stdout, and stderr with pipe0, pipe1, and pipe2
    check(dup2(pipe0[0], 0), 0);
    check(dup2(pipe1[1], 1), 1);
    check(dup2(pipe2[1], 2), 2);

    // Close old fds of pipe0, pipe1, and pipe2
    check(close(pipe0[0]));
    check(close(pipe1[1]));
    check(close(pipe2[1]));

    // Invoke sodac
    check(execlp("/bin/sh", "/bin/sh", "-c",
          "python3 $(which sodac) --xocl-kernel - -", nullptr));
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

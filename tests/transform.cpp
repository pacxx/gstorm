#include <action.h>
#include <view.h>
#include <vector>
#include <algorithm>
#include <iostream>


using namespace gstorm;
using namespace std;
using namespace pacxx::v2;

template<typename T>
void __print(const T& rng) {
  for (auto v : rng)
    cout << v;
  cout << endl;
}


int main(int argc, char *argv[])
{

  size_t count = 100;
  if (argc >= 2)
    count = stoi(argv[1]);

  cout << "test for " << count << " elements" << endl;
  vector<int> a(count), b(count), c(count), d(count), e(1);
  fill(a.begin(), a.end(), 1);

  auto view1 = range::vector(a);
//  auto view2 = range::vector(b);
  //  auto ref = view::repeat<const decltype(e)&>(e);
  //  auto ref = view::scalar<int>(5);


//  auto gcopy = a | gpu::copy | gpu::action::transform([](auto in) { return in * 2; });

//  std::vector<int> copy_from_a  = a | gpu::copy | gpu::action::transform([](auto in) { return in * 2; });
  std::vector<int> copy_from_a2 = a | gpu::action::transform([](auto in) { return in * 2; });
//  __print(copy_from_a);
  __print(copy_from_a2);

  auto t1 = view::transform(view1, [](auto in) { return in * 2; });

  vector<int> x = view::transform(view1, [](auto in) { return in * 3; });

//  action::transform(t1, b);

//  __print(x);
//  __print(b);




//  auto op = [](auto x, auto y, const auto& v) { return v[0] * x + y; };
//
//  auto forward_tpl = [=](auto&& tpl) { return meta::apply(op, tpl); };
//
//  auto zip1 = view::zip(a, b, ref);
//
//  action::transform(zip1, c, forward_tpl);
//
//  action::gpu::transform(zip1, d, forward_tpl);
//
//  for (auto v : c) cout << v;
//  cout << endl;
//
//  auto& exec = get_executor();
//  auto& md = exec.mm().translateVector(d);
//  md.download(d.data(), d.size() * sizeof(int), 256);
//
//  for (auto v : d) cout << v;
//  cout << endl;

}

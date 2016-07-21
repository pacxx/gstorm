#include <action.h>
#include <view.h>
#include <vector>
#include <algorithm>
#include <iostream>


using namespace gstorm;
using namespace std;

int main()
{
  vector<int> a(100), b(100), c(100), d(100);
  fill(a.begin(), a.end(), 1);

  auto view1 = view::vector(a);
  auto view2 = view::vector(b);
  auto sa = view::scalar(5);

  action::transform(view1, view2);

  auto zip1 = view::zip(a, b, sa);
  auto saxpy = [](auto x, auto y, auto a){ return a * x + y; };

  action::transform(zip1, c, [=](auto&& tpl){ return meta::apply(saxpy, tpl);});

//  action::gpu::transform(zip1, d, [](auto&& tpl){ return get<0>(tpl) + get<1>(tpl);});

  for (auto v : c) cout << v;
  cout << endl;

  for (auto v : d) cout << v;
  cout << endl;

}
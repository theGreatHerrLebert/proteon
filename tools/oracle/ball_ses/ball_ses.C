// Minimal BALL SES oracle: read an xyzr sphere file (x y z radius per line),
// compute BALL's analytic SES area+volume at a given probe radius, print them.
// Feeds the SAME spheres proteon's mesher uses (radii match exactly).
#include <BALL/KERNEL/fragment.h>
#include <BALL/KERNEL/atom.h>
#include <BALL/STRUCTURE/analyticalSES.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace BALL;

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: ball_ses spheres.xyzr [probe=1.4]\n"; return 2; }
  float probe = (argc >= 3) ? std::atof(argv[2]) : 1.4f;
  std::ifstream in(argv[1]);
  if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }

  Fragment frag;
  std::vector<Atom*> atoms;
  double x, y, z, r;
  while (in >> x >> y >> z >> r) {
    Atom* a = new Atom();
    a->setPosition(Vector3(x, y, z));
    a->setRadius(r);
    frag.insert(*a);
    atoms.push_back(a);
  }
  std::cerr << "loaded " << atoms.size() << " spheres, probe=" << probe << "\n";

  try {
    float area = calculateSESArea(frag, probe);
    float vol  = calculateSESVolume(frag, probe);
    std::cout << "OK area=" << area << " volume=" << vol << "\n";
  } catch (Exception::GeneralException& e) {
    std::cout << "FAIL " << e.getName() << ": " << e.getMessage() << "\n";
    return 1;
  } catch (...) {
    std::cout << "FAIL unknown\n";
    return 1;
  }
  return 0;
}

# -*- encoding: utf-8 -*-
"""
@Description:       :
uff torsion angle term 
return its energy and gradient 

@Date     :2021/06/04 11:11:25
@Author      :likun.yang
"""
import numpy as np

from uff_angle import *


def is_not_equal(ab, cd):
    ab = np.array(ab)
    cd = np.array(cd)
    ab = np.sort(ab)
    cd = np.sort(cd)
    return np.all(ab != cd)


def is_equal(ab, cd):
    ab = np.array(ab)
    cd = np.array(cd)
    ab = np.sort(ab)
    cd = np.sort(cd)
    return np.all(ab == cd)


def _sorted_pair(ab):
    a = ab[0]
    b = ab[1]
    if b < a:
        tmp = b
        b = a
        a = tmp
    return (a, b)


def get_torsion_list(angle_list):
    """
    The torsional terms for two bonds IJ and KL
    connected via a common bond JK

    make sure nodes in the middle beg - node1 - node2 - end
    """

    torsion_list = []
    n_angle = len(angle_list)
    for i in range(n_angle):
        ab = angle_list[i][:2]
        cb = angle_list[i][1:][::-1]  ## [::-1] make sure node in the middle
        for j in range(i + 1, n_angle):
            de = angle_list[j][:2]
            fe = angle_list[j][1:][::-1]
            if is_equal(ab, de) and is_not_equal(cb, fe):
                torsion_list.append(cb + fe)
            elif is_equal(ab, fe) and is_not_equal(cb, de):
                torsion_list.append(cb + de[::-1])
            elif is_equal(cb, de) and is_not_equal(ab, fe):
                torsion_list.append(ab + fe[::-1])
            elif is_equal(cb, fe) and is_not_equal(ab, de):
                torsion_list.append(ab + de[::-1])
    return torsion_list


"""
this is how rdkit find torsional angle

# void addTorsions(const ROMol & mol, const AtomicParamVect & params,
#                  ForceFields: : ForceField * field,
#                  const std: : string & torsionBondSmarts) {
#     PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");
#     PRECONDITION(field, "bad forcefield");

#     // find all of the torsion bonds:
#     std:: vector < MatchVectType > matchVect;
#     const ROMol * defaultQuery = DefaultTorsionBondSmarts:: query();
#     const ROMol * query = (torsionBondSmarts == DefaultTorsionBondSmarts:: string())
#                            ? defaultQuery: SmartsToMol(torsionBondSmarts);
#     TEST_ASSERT(query);
#     unsigned int nHits = SubstructMatch(mol, *query, matchVect);
#     if (query != defaultQuery) {
#         delete query;}

#     for (unsigned int i=0; i < nHits; i++) {
#         MatchVectType match = matchVect[i];
#         TEST_ASSERT(match.size() == 2);
#         int idx1 = match[0].second;
#         int idx2 = match[1].second;
#         if (!params[idx1] | | !params[idx2]) {
#             continue; }
#         const Bond * bond = mol.getBondBetweenAtoms(idx1, idx2);
#         std:: vector < TorsionAngleContrib * > contribsHere;
#         TEST_ASSERT(bond);
#         const Atom * atom1 = mol.getAtomWithIdx(idx1);
#         const Atom * atom2 = mol.getAtomWithIdx(idx2);

#         if ((atom1 -> getHybridization() == Atom:: SP2 | |
#              atom1 -> getHybridization() == Atom:: SP3) & &
#             (atom2 -> getHybridization() == Atom:: SP2 | |
#              atom2 -> getHybridization() == Atom: : SP3)) {
#             ROMol:: OEDGE_ITER beg1, end1;
#             boost:: tie(beg1, end1) = mol.getAtomBonds(atom1);
#             while (beg1 != end1) {
#                 const Bond * tBond1 = mol[*beg1];
#                 if (tBond1 != bond) {
#                     int bIdx = tBond1 -> getOtherAtomIdx(idx1);
#                     ROMol: : OEDGE_ITER beg2, end2;
#                     boost: : tie(beg2, end2) = mol.getAtomBonds(atom2);
#                     while (beg2 != end2) {
#                         const Bond * tBond2 = mol[*beg2];
#                         if (tBond2 != bond & & tBond2 != tBond1) {
#                             int eIdx = tBond2 -> getOtherAtomIdx(idx2);
#                             // make sure this isn't a three-membered ring:
#                             if (eIdx != bIdx) {
#                                 // we now have a torsion involving atoms(bonds):
#                                 // bIdx - (tBond1) - idx1 - (bond) - idx2 - (tBond2) - eIdx
#                                 TorsionAngleContrib * contrib;

#                                 // if either of the end atoms is SP2 hybridized, set a flag
#                                 // here.
#                                 bool hasSP2 = false;
#                                 if (mol.getAtomWithIdx(bIdx) -> getHybridization() == Atom:: SP2 | |
#                                     mol.getAtomWithIdx(eIdx) -> getHybridization() == Atom: : SP2) {
#                                     hasSP2 = true;}
#                                 // std:: cout << "Torsion: " << bIdx << "-" << idx1 << "-" <<
#                                 // idx2 << "-" << eIdx << std:: endl;
#                                 // if(okToIncludeTorsion(mol, bond, bIdx, idx1, idx2, eIdx)){
#                                     // std:: cout << "  INCLUDED" << std: : endl;
#                                     contrib = new TorsionAngleContrib(
#                                         field, bIdx, idx1, idx2, eIdx, bond -> getBondTypeAsDouble(),
#                                         atom1 -> getAtomicNum(), atom2 -> getAtomicNum(),
#                                         atom1 -> getHybridization(), atom2 -> getHybridization(),
#                                         params[idx1], params[idx2], hasSP2);
#                                     field -> contribs().push_back(ForceFields:: ContribPtr(contrib));
#                                     contribsHere.push_back(contrib);
#                                     //}
#                             }
#                         }
#                         beg2++;}
#                 }
#                 beg1++; }
#         }
#         // now divide the force constant for each contribution to the torsion energy
#         // about this bond by the number of contribs about this bond:
#         for (auto chI=contribsHere.begin(); chI != contribsHere.end(); ++chI) {
#             (*chI) -> scaleForceConstant(contribsHere.size()); }
#     }
# }
"""

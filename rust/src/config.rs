/*
 * @Author: your name
 * @Date: 2021-05-26 10:49:23
 * @LastEditTime: 2021-05-28 17:17:04
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /rust/src/config.rs
 */
use std::{collections::HashMap, sync::Mutex};

use log::warn;
use once_cell::sync::Lazy;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const VDW_RADII: Lazy<Mutex<HashMap<&str, f64>>> = Lazy::new(|| {
    let m = [
        ("C", 1.7),
        ("CA", 1.7),
        ("CB", 1.7),
        ("N", 1.55),
        ("O", 1.52),
        ("CD", 1.7),
        ("CG", 1.7),
        ("CZ", 1.7),
        ("NE", 1.55),
        ("NH1", 1.55),
        ("NH2", 1.55),
        ("ND2", 1.55),
        ("OD1", 1.52),
        ("OD2", 1.52),
        ("SG", 1.8),
        ("NE2", 1.55),
        ("OE1", 1.52),
        ("OE2", 1.52),
        ("CD2", 1.7),
        ("CE1", 1.7),
        ("ND1", 1.55),
        ("CD1", 1.7),
        ("CG1", 1.7),
        ("CG2", 1.7),
        ("OXT", 1.52),
        ("CE", 1.7),
        ("NZ", 1.55),
        ("SD", 1.8),
        ("CE2", 1.7),
        ("OG", 1.52),
        ("OG1", 1.52),
        ("CE3", 1.7),
        ("CH2", 1.7),
        ("CZ2", 1.7),
        ("CZ3", 1.7),
        ("NE1", 1.55),
        ("OH", 1.52),
        ("H1", 1.2),
        ("H2", 1.2),
        ("P", 1.8),
        ("O1P", 1.52),
        ("O2P", 1.52),
        ("O3P", 1.52),
        ("F", 1.47),
        ("MN", 0.8),
        ("NA", 0.8),
        ("ZN", 0.8),
    ]
    .iter()
    .cloned()
    .collect();

    Mutex::new(m)
});

/// 残基_原子 ==> 疏水性
const ATOMS_HYDROPHOBICITY: Lazy<Mutex<HashMap<&str, i32>>> = Lazy::new(|| {
    let m = [
        ("ALA_C", 12),
        ("ALA_CA", 12),
        ("ALA_CB", 12),
        ("ALA_N", -6),
        ("ALA_O", -6),
        ("ARG_C", 12),
        ("ARG_CA", 12),
        ("ARG_CB", 12),
        ("ARG_CD", 12),
        ("ARG_CG", 12),
        ("ARG_CZ", 12),
        ("ARG_N", -6),
        ("ARG_NE", -6),
        ("ARG_NH1", -10),
        ("ARG_NH2", -10),
        ("ARG_O", -6),
        ("ASN_C", 12),
        ("ASN_CA", 12),
        ("ASN_CB", 12),
        ("ASN_CG", 12),
        ("ASN_N", -6),
        ("ASN_ND2", -6),
        ("ASN_O", -6),
        ("ASN_OD1", -6),
        ("ASP_C", 12),
        ("ASP_CA", 12),
        ("ASP_CB", 12),
        ("ASP_CG", 12),
        ("ASP_N", -6),
        ("ASP_O", -6),
        ("ASP_OD1", -18),
        ("ASP_OD2", -18),
        ("CYS_C", 12),
        ("CYS_CA", 12),
        ("CYS_CB", 12),
        ("CYS_N", -6),
        ("CYS_O", -6),
        ("CYS_SG", 36),
        ("GLN_C", 12),
        ("GLN_CA", 12),
        ("GLN_CB", 12),
        ("GLN_CD", 12),
        ("GLN_CG", 12),
        ("GLN_N", -6),
        ("GLN_NE2", -6),
        ("GLN_O", -6),
        ("GLN_OE1", -6),
        ("GLU_C", 12),
        ("GLU_CA", 12),
        ("GLU_CB", 12),
        ("GLU_CD", 12),
        ("GLU_CG", 12),
        ("GLU_N", -6),
        ("GLU_O", -6),
        ("GLU_OE1", -18),
        ("GLU_OE2", -18),
        ("GLY_C", 12),
        ("GLY_CA", 12),
        ("GLY_N", -6),
        ("GLY_O", -6),
        ("HIS_C", 12),
        ("HIS_CA", 12),
        ("HIS_CB", 12),
        ("HIS_CD2", 12),
        ("HIS_CE1", 12),
        ("HIS_CG", 12),
        ("HIS_N", -6),
        ("HIS_ND1", -6),
        ("HIS_NE2", -6),
        ("HIS_O", -6),
        ("ILE_C", 12),
        ("ILE_CA", 12),
        ("ILE_CB", 12),
        ("ILE_CD1", 12),
        ("ILE_CG1", 12),
        ("ILE_CG2", 12),
        ("ILE_N", -6),
        ("ILE_O", -6),
        ("LEU_C", 12),
        ("LEU_CA", 12),
        ("LEU_CB", 12),
        ("LEU_CD1", 12),
        ("LEU_CD2", 12),
        ("LEU_CG", 12),
        ("LEU_N", -6),
        ("LEU_O", -6),
        ("LEU_OXT", -6),
        ("LYS_C", 12),
        ("LYS_CA", 12),
        ("LYS_CB", 12),
        ("LYS_CD", 12),
        ("LYS_CE", 12),
        ("LYS_CG", 12),
        ("LYS_N", -6),
        ("LYS_NZ", -19),
        ("LYS_O", -6),
        ("MET_C", 12),
        ("MET_CA", 12),
        ("MET_CB", 12),
        ("MET_CE", 12),
        ("MET_CG", 12),
        ("MET_N", -6),
        ("MET_O", -6),
        ("MET_SD", 36),
        ("PHE_C", 12),
        ("PHE_CA", 12),
        ("PHE_CB", 12),
        ("PHE_CD1", 12),
        ("PHE_CD2", 12),
        ("PHE_CE1", 12),
        ("PHE_CE2", 12),
        ("PHE_CG", 12),
        ("PHE_CZ", 12),
        ("PHE_N", -6),
        ("PHE_O", -6),
        ("PRO_C", 12),
        ("PRO_CA", 12),
        ("PRO_CB", 12),
        ("PRO_CD", 12),
        ("PRO_CG", 12),
        ("PRO_N", -6),
        ("PRO_O", -6),
        ("SER_C", 12),
        ("SER_CA", 12),
        ("SER_CB", 12),
        ("SER_N", -6),
        ("SER_O", -6),
        ("SER_OG", -6),
        ("THR_C", 12),
        ("THR_CA", 12),
        ("THR_CB", 12),
        ("THR_CG2", 12),
        ("THR_N", -6),
        ("THR_O", -6),
        ("THR_OG1", -6),
        ("THR_OXT", -6),
        ("TRP_C", 12),
        ("TRP_CA", 12),
        ("TRP_CB", 12),
        ("TRP_CD1", 12),
        ("TRP_CD2", 12),
        ("TRP_CE2", 12),
        ("TRP_CE3", 12),
        ("TRP_CG", 12),
        ("TRP_CH2", 12),
        ("TRP_CZ2", 12),
        ("TRP_CZ3", 12),
        ("TRP_N", -6),
        ("TRP_NE1", -6),
        ("TRP_O", -6),
        ("TYR_C", 12),
        ("TYR_CA", 12),
        ("TYR_CB", 12),
        ("TYR_CD1", 12),
        ("TYR_CD2", 12),
        ("TYR_CE1", 12),
        ("TYR_CE2", 12),
        ("TYR_CG", 12),
        ("TYR_CZ", 12),
        ("TYR_N", -6),
        ("TYR_O", -6),
        ("TYR_OH", -6),
        ("VAL_C", 12),
        ("VAL_CA", 12),
        ("VAL_CB", 12),
        ("VAL_CG1", 12),
        ("VAL_CG2", 12),
        ("VAL_N", -6),
        ("VAL_O", -6),
        ("VAL_OXT", -6),
        ("NA_NA", -12),
        ("WAT_O", -6),
        ("WAT_H1", 0),
        ("WAT_H2", 0),
        ("HOH_O", -6),
        ("PTR_N", -6),
        ("PTR_CA", 12),
        ("PTR_C", 12),
        ("PTR_O", -6),
        ("PTR_CB", 12),
        ("PTR_CG", 12),
        ("PTR_CD1", 12),
        ("PTR_CD2", 12),
        ("PTR_CE1", 12),
        ("PTR_CE2", 12),
        ("PTR_CZ", 12),
        ("PTR_OH", -6),
        ("PTR_P", 36),
        ("PTR_O1P", -9),
        ("PTR_O2P", -9),
        ("PTR_O3P", -9),
        ("SEP_N", -6),
        ("SEP_CA", 12),
        ("SEP_C", 12),
        ("SEP_O", -6),
        ("SEP_CB", 12),
        ("SEP_OG", -6),
        ("SEP_P", 36),
        ("SEP_O1P", -9),
        ("SEP_O2P", -9),
        ("SEP_O3P", -9),
        ("TPO_N", -6),
        ("TPO_CA", 12),
        ("TPO_C", 12),
        ("TPO_O", -6),
        ("TPO_CB", 12),
        ("TPO_CG2", -6),
        ("TPO_OG1", -6),
        ("TPO_P", 36),
        ("TPO_O1P", -9),
        ("TPO_O2P", -9),
        ("TPO_O3P", -9),
        ("MN_MN", -12),
    ]
    .iter()
    .cloned()
    .collect();

    Mutex::new(m)
});

pub fn get_atom_hpd(atom: &str, resn: &str, mm: &HashMap<&str, i32>) -> f64 {
    if atom == "OXT" {
        return -6.;
    } else if atom == "NA" {
        return -12.;
    }

    let key = format!("{}_{}", resn, atom);
    let f = mm.get(key.as_str()).map_or_else(
        || {
            warn!("key : {} not found atom hpd map", &key);
            0.
        },
        |p| *p as f64,
    );
    f
}
pub fn get_radii(key: &str) -> f64 {
    let f = VDW_RADII.lock().unwrap().get(key).map_or_else(
        || {
            warn!("key : {} not found vdw radii map", &key);
            0.
        },
        |p| *p,
    );
    f
}

pub fn get_vdw_radii(elements: Option<&Vec<&str>>, pr: f64, i: usize) -> f64 {
    match elements {
        Some(e) => get_radii(e[i]) + pr,
        None => pr,
    }
}

pub fn get_vdw_vec(elements: Option<&Vec<&str>>, radis_v: &mut Vec<f64>) {
    let mm = VDW_RADII.lock().unwrap().clone();
    let get_vdw_radii = move |elements: Option<&Vec<&str>>, i: usize| {
        if let Some(e) = elements {
            mm.get(e[i]).unwrap() + 0.
        } else {
            0.
        }
    };

    radis_v.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = get_vdw_radii(elements, i);
    });
}

pub fn get_hdp_vec(elements: Option<&Vec<&str>>, resns: &Vec<&str>, hdp_v: &mut Vec<f64>) {
    let mm = ATOMS_HYDROPHOBICITY.lock().unwrap().clone();
    let get_atom_hdp = move |elements: Option<&Vec<&str>>, i: usize| {
        if let Some(e) = elements {
            if e.get(i).is_some() {
                let atom = e[i];
                let resn = resns[i];
                get_atom_hpd(atom, resn, &mm)
            } else {
                0.
            }
        } else {
            0.
        }
    };

    hdp_v.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = get_atom_hdp(elements, i);
    });
}

pub fn init_config() {
    let r = log4rs::init_file("config/log4rs.yaml", Default::default());

    if r.is_err() {
        let _ = log4rs::init_file("rust/config/log4rs.yaml", Default::default());
    }
}

mod tests {

    #[test]
    fn test_vdw() {
        super::init_config();
        assert_eq!(super::get_radii("C"), 1.7f64);
        assert_ne!(super::get_radii("O"), 1.7f64);
        assert_eq!(super::get_radii("1"), 0.);

        let mm = super::ATOMS_HYDROPHOBICITY.lock().unwrap().clone();

        assert_eq!(super::get_atom_hpd("atom", "", &mm), 0.);
        assert_eq!(super::get_atom_hpd("OXT", "", &mm), -6.);
        assert_eq!(super::get_atom_hpd("NA", "", &mm), -12.);
        assert_eq!(super::get_atom_hpd("C", "ARG", &mm), 12.);
        assert_eq!(super::get_atom_hpd("OD2", "ASP", &mm), -18.);
        assert_eq!(super::get_atom_hpd("CA", "CYS", &mm), 12.);
        assert_eq!(super::get_atom_hpd("NE2", "GLN", &mm), -6.);
    }
}
